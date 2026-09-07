"""Set-piece delivery and landing-point candidates from ball trajectories."""

from __future__ import annotations

import math

from ..base import TacticalAnalyzer, register_analyzer
from ..models import HALF_LENGTH, HALF_WIDTH, AnalysisContext, AnalyzerOutput, Evidence, Finding, TimelineEvent
from .events import EventTimelineAnalyzer, attack_direction, ball_point, nearest_ball_player


def landing_zone(x: float, y: float, direction: int) -> str:
    if abs(x) > HALF_LENGTH + 2.0 or abs(y) > HALF_WIDTH + 2.0:
        return "out_of_bounds"
    progress = x * direction
    if progress >= HALF_LENGTH - 16.5 and abs(y) <= 20.16:
        return "penalty_area"
    if progress >= HALF_LENGTH / 3.0 and 7.0 <= abs(y) <= 18.0:
        return "attacking_half_space"
    if abs(y) >= HALF_WIDTH / 2.0:
        return "wide_channel"
    return "central_channel"


ZONE_LABELS = {
    "out_of_bounds": "场外区域",
    "penalty_area": "禁区",
    "attacking_half_space": "进攻肋部",
    "wide_channel": "边路",
    "central_channel": "中路",
}

TYPE_LABELS = {
    "corner": "角球",
    "free_kick": "任意球",
    "goal_kick": "球门球",
    "throw_in": "界外球",
    "kickoff": "中圈开球",
    "penalty": "点球",
}


def classify_set_piece(origin_x: float, origin_y: float, team: str | None, direction: int) -> str:
    """Classify a restart conservatively from pitch geometry and team direction."""
    if abs(origin_x) >= HALF_LENGTH - 7.0 and abs(origin_y) >= HALF_WIDTH - 7.0:
        return "corner"
    if math.hypot(origin_x, origin_y) <= 6.0:
        return "kickoff"
    if abs(abs(origin_x) - 41.5) <= 3.0 and abs(origin_y) <= 4.0 and origin_x * direction > 0:
        return "penalty"
    if abs(origin_y) >= HALF_WIDTH - 2.5:
        return "throw_in"
    if team in {"left", "right"} and abs(origin_x) >= HALF_LENGTH - 10.0 and abs(origin_y) <= 14.0:
        if origin_x * direction < 0:
            return "goal_kick"
    return "free_kick"


@register_analyzer("set_piece_delivery")
class SetPieceDeliveryAnalyzer(TacticalAnalyzer):
    priority = "P1"
    description_zh = "识别角球/任意球开出并估计首次稳定落点和落点控制"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        stationary_speed = float(self.config.get("stationary_speed_mps", 2.5))
        restart_speed = float(self.config.get("restart_speed_mps", 6.0))
        minimum_stationary = float(self.config.get("minimum_stationary_sec", 0.7))
        landing_window = float(self.config.get("landing_window_sec", 7.0))
        landing_radius = float(self.config.get("landing_control_radius_m", 6.0))
        cooldown = float(self.config.get("candidate_cooldown_sec", 8.0))
        maximum_gap = float(self.config.get("maximum_ball_gap_sec", 0.6))

        ball_samples: list[dict[str, object]] = []
        previous: tuple[float, float, float] | None = None
        for index, frame in enumerate(context.frames):
            point = ball_point(frame)
            if point is None:
                continue
            speed = None
            if previous is not None:
                previous_time, previous_x, previous_y = previous
                dt = frame.time_sec - previous_time
                if 0 < dt <= maximum_gap:
                    speed = math.hypot(point[0] - previous_x, point[1] - previous_y) / dt
            ball_samples.append(
                {"context_index": index, "frame": frame, "x": point[0], "y": point[1], "speed": speed}
            )
            previous = (frame.time_sec, point[0], point[1])

        deliveries: list[dict[str, object]] = []
        stationary_start: float | None = None
        last_candidate = -999.0
        for sample_index, sample in enumerate(ball_samples):
            frame = sample["frame"]
            speed = sample["speed"]
            assert hasattr(frame, "time_sec")
            if speed is None:
                stationary_start = None
                continue
            if float(speed) <= stationary_speed:
                stationary_start = stationary_start if stationary_start is not None else frame.time_sec
                continue
            stationary_duration = frame.time_sec - stationary_start if stationary_start is not None else 0.0
            if (
                float(speed) < restart_speed
                or stationary_duration < minimum_stationary
                or frame.time_sec - last_candidate < cooldown
                or sample_index == 0
            ):
                if float(speed) > stationary_speed:
                    stationary_start = None
                continue

            origin_sample = ball_samples[sample_index - 1]
            origin_x, origin_y = float(origin_sample["x"]), float(origin_sample["y"])
            taker, taker_distance = nearest_ball_player(frame, 10.0)
            team = taker.team if taker else None
            direction = attack_direction(frame, team) if team in {"left", "right"} else (1 if origin_x < 0 else -1)
            set_piece_type = classify_set_piece(origin_x, origin_y, team, direction)

            landing = self._find_landing(
                ball_samples,
                sample_index,
                origin_x,
                origin_y,
                landing_window,
                landing_radius,
                taker.track_id if taker else None,
            )
            if landing is None:
                stationary_start = None
                continue
            landing_team = landing.get("landing_team")
            delivery = {
                "set_piece_type": set_piece_type,
                "restart_sec": round(frame.time_sec, 3),
                "restart_frame": frame.frame,
                "team": team,
                "taker_track_id": taker.track_id if taker else None,
                "taker_distance_m": round(taker_distance, 3) if taker_distance is not None else None,
                "origin_x": round(origin_x, 3),
                "origin_y": round(origin_y, 3),
                "restart_speed_mps": round(float(speed), 3),
                **landing,
                "landing_zone": landing_zone(float(landing["landing_x"]), float(landing["landing_y"]), direction),
                "retained_by_taking_team": bool(team is not None and landing_team == team),
            }
            deliveries.append(delivery)
            last_candidate = frame.time_sec
            stationary_start = None

        # The base detector has a dedicated corner-state machine. Use those
        # events as stronger restart evidence when the generic stop detector misses.
        base_events = EventTimelineAnalyzer(self.config).analyze(context).events
        detected_times = [float(item["restart_sec"]) for item in deliveries]
        for corner in (event for event in base_events if event.event_type == "corner_candidate"):
            if any(abs(corner.time_sec - old) < 3.0 for old in detected_times) or not ball_samples:
                continue
            sample_index = min(
                range(len(ball_samples)),
                key=lambda index: abs(ball_samples[index]["frame"].time_sec - corner.time_sec),
            )
            origin_sample = ball_samples[max(0, sample_index - 1)]
            origin_x, origin_y = float(origin_sample["x"]), float(origin_sample["y"])
            frame = ball_samples[sample_index]["frame"]
            landing = self._find_landing(
                ball_samples,
                sample_index,
                origin_x,
                origin_y,
                landing_window,
                landing_radius,
                corner.actor_track_id,
            )
            if landing is None:
                continue
            team = corner.team
            direction = attack_direction(frame, team) if team in {"left", "right"} else (1 if origin_x < 0 else -1)
            landing_team = landing.get("landing_team")
            deliveries.append(
                {
                    "set_piece_type": "corner",
                    "restart_sec": round(corner.time_sec, 3),
                    "restart_frame": corner.frame,
                    "team": team,
                    "taker_track_id": corner.actor_track_id,
                    "taker_distance_m": None,
                    "origin_x": round(origin_x, 3),
                    "origin_y": round(origin_y, 3),
                    "restart_speed_mps": corner.metrics.get("restart_speed_mps"),
                    **landing,
                    "landing_zone": landing_zone(float(landing["landing_x"]), float(landing["landing_y"]), direction),
                    "retained_by_taking_team": bool(team is not None and landing_team == team),
                    "source_event_id": corner.event_id,
                }
            )
            detected_times.append(corner.time_sec)

        deliveries.sort(key=lambda item: float(item["restart_sec"]))

        events: list[TimelineEvent] = []
        findings: list[Finding] = []
        shot_events = [event for event in base_events if event.event_type == "shot_candidate"]
        for delivery in deliveries:
            followup = next(
                (
                    shot for shot in shot_events
                    if float(delivery["restart_sec"]) < shot.time_sec <= float(delivery["restart_sec"]) + 10.0
                    and (delivery["team"] is None or shot.team == delivery["team"])
                ),
                None,
            )
            delivery["shot_within_10s"] = followup is not None
            delivery["followup_shot_event_id"] = followup.event_id if followup else None
            delivery["followup_result"] = "shot_candidate" if followup else "no_shot_detected"

        for index, delivery in enumerate(deliveries, 1):
            type_label = TYPE_LABELS.get(str(delivery["set_piece_type"]), "定位球")
            team_label = {"left": "左队", "right": "右队"}.get(delivery["team"], "未知球队")
            zone_label = ZONE_LABELS.get(str(delivery["landing_zone"]), str(delivery["landing_zone"]))
            controlled = delivery["landing_track_id"] is not None
            confidence = min(0.9, 0.52 + (0.14 if controlled else 0.0) + min(float(delivery["flight_distance_m"]) / 100.0, 0.16))
            event = TimelineEvent(
                event_id=f"set-piece-{index:03d}",
                event_type="set_piece_delivery_candidate",
                frame=int(delivery["restart_frame"]),
                time_sec=float(delivery["restart_sec"]),
                team=delivery["team"] if delivery["team"] in {"left", "right"} else None,
                actor_track_id=delivery["taker_track_id"],
                target_track_id=delivery["landing_track_id"],
                confidence=round(confidence, 4),
                label_zh=f"{type_label}落点候选",
                metrics=delivery,
                event_subtype=str(delivery["set_piece_type"]),
                outcome=(
                    "retained" if delivery["retained_by_taking_team"]
                    else "opponent_control" if delivery["landing_team"] is not None
                    else "unknown"
                ),
                start_x=float(delivery["origin_x"]),
                start_y=float(delivery["origin_y"]),
                end_x=float(delivery["landing_x"]),
                end_y=float(delivery["landing_y"]),
                evidence_start_sec=max(0.0, float(delivery["restart_sec"]) - 4.0),
                evidence_end_sec=float(delivery["landing_sec"]) + 4.0,
                related_event_ids=[delivery["followup_shot_event_id"]] if delivery["followup_shot_event_id"] else [],
            )
            events.append(event)
            findings.append(
                Finding(
                    finding_id=f"set-piece-landing-{index:03d}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="set_piece_landing_candidate",
                    title_zh=f"{delivery['restart_sec']:.2f}秒 {team_label}{type_label}落向{zone_label}",
                    summary_zh=(
                        f"估计飞行{float(delivery['flight_distance_m']):.1f}米、"
                        f"{float(delivery['flight_duration_sec']):.1f}秒；"
                        f"{'开球队控制到落点' if delivery['retained_by_taking_team'] else '落点控制未确认或由对手获得'}；"
                        f"{'10秒内形成射门候选' if delivery['shot_within_10s'] else '10秒内未检测到射门'}。"
                    ),
                    confidence=event.confidence,
                    start_sec=max(0.0, event.time_sec - 3.0),
                    end_sec=float(delivery["landing_sec"]) + 3.0,
                    metrics=delivery,
                    evidence=[Evidence(event.frame, event.time_sec, "足球静止后加速开出，并追踪至首次稳定控制。", delivery)],
                    limitations_zh=["任意球是静止重启代理，界外球、门球或短暂停球可能混入，需结合原视频复核。"],
                )
            )

        team_summary = {}
        for team in ("left", "right"):
            team_items = [item for item in deliveries if item["team"] == team]
            team_summary[team] = {
                "deliveries": len(team_items),
                "corners": sum(item["set_piece_type"] == "corner" for item in team_items),
                "free_kicks": sum(item["set_piece_type"] == "free_kick" for item in team_items),
                "goal_kicks": sum(item["set_piece_type"] == "goal_kick" for item in team_items),
                "throw_ins": sum(item["set_piece_type"] == "throw_in" for item in team_items),
                "kickoffs": sum(item["set_piece_type"] == "kickoff" for item in team_items),
                "penalties": sum(item["set_piece_type"] == "penalty" for item in team_items),
                "penalty_area_landings": sum(item["landing_zone"] == "penalty_area" for item in team_items),
                "retained_landings": sum(bool(item["retained_by_taking_team"]) for item in team_items),
                "shots_within_10s": sum(bool(item["shot_within_10s"]) for item in team_items),
            }
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "stationary_restart_to_first_control_v1", "teams": team_summary, "deliveries": deliveries},
            findings=findings,
            events=events,
            warnings=["定位球类型和落点控制均为二维足球轨迹代理，不包含球的高度、旋转和真实第一落点。"],
        )

    @staticmethod
    def _find_landing(
        samples: list[dict[str, object]],
        restart_index: int,
        origin_x: float,
        origin_y: float,
        window_sec: float,
        control_radius: float,
        taker_track_id: int | None,
    ) -> dict[str, object] | None:
        restart_frame = samples[restart_index]["frame"]
        last_sample = None
        for sample in samples[restart_index + 1:]:
            frame = sample["frame"]
            if frame.time_sec - restart_frame.time_sec > window_sec:
                break
            last_sample = sample
            distance = math.hypot(float(sample["x"]) - origin_x, float(sample["y"]) - origin_y)
            if distance < 6.0 or frame.time_sec - restart_frame.time_sec < 0.15:
                continue
            receiver, receiver_distance = nearest_ball_player(frame, control_radius)
            if receiver and receiver.track_id != taker_track_id:
                return {
                    "landing_sec": round(frame.time_sec, 3),
                    "landing_frame": frame.frame,
                    "landing_x": round(float(sample["x"]), 3),
                    "landing_y": round(float(sample["y"]), 3),
                    "flight_distance_m": round(distance, 3),
                    "flight_duration_sec": round(frame.time_sec - restart_frame.time_sec, 3),
                    "landing_team": receiver.team,
                    "landing_track_id": receiver.track_id,
                    "landing_control_distance_m": round(receiver_distance, 3) if receiver_distance is not None else None,
                    "landing_source": "first_stable_control",
                }
        if last_sample is None:
            return None
        frame = last_sample["frame"]
        distance = math.hypot(float(last_sample["x"]) - origin_x, float(last_sample["y"]) - origin_y)
        if distance < 6.0:
            return None
        return {
            "landing_sec": round(frame.time_sec, 3),
            "landing_frame": frame.frame,
            "landing_x": round(float(last_sample["x"]), 3),
            "landing_y": round(float(last_sample["y"]), 3),
            "flight_distance_m": round(distance, 3),
            "flight_duration_sec": round(frame.time_sec - restart_frame.time_sec, 3),
            "landing_team": None,
            "landing_track_id": None,
            "landing_control_distance_m": None,
            "landing_source": "trajectory_endpoint",
        }
