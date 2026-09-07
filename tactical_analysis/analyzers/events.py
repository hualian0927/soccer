"""Evidence-based event timeline with constrained possession smoothing."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from ..base import TacticalAnalyzer, register_analyzer
from ..models import (
    HALF_LENGTH,
    HALF_WIDTH,
    AnalysisContext,
    AnalyzerOutput,
    Evidence,
    Finding,
    FrameState,
    Observation,
    TimelineEvent,
)


@dataclass(frozen=True)
class PossessionSample:
    frame: int
    time_sec: float
    team: str | None
    track_id: int | None
    distance_m: float | None
    confidence: float


class ConstrainedPossessionDecoder:
    """Small online sequence decoder that suppresses impossible one-frame switches."""

    def __init__(self, control_radius: float, confirm_samples: int, release_samples: int) -> None:
        self.control_radius = control_radius
        self.confirm_samples = max(1, confirm_samples)
        self.release_samples = max(1, release_samples)
        self.active: tuple[str, int] | None = None
        self.pending: tuple[str, int] | None = None
        self.pending_count = 0
        self.missing_count = 0

    def update(self, frame: FrameState) -> PossessionSample:
        candidate, distance = nearest_ball_player(frame, self.control_radius)
        raw = (candidate.team, candidate.track_id) if candidate and candidate.team else None

        if raw is None:
            self.missing_count += 1
            if self.missing_count >= self.release_samples:
                self.active = None
            self.pending = None
            self.pending_count = 0
        elif raw == self.active:
            self.missing_count = 0
            self.pending = None
            self.pending_count = 0
        else:
            self.missing_count = 0
            if raw == self.pending:
                self.pending_count += 1
            else:
                self.pending = raw
                self.pending_count = 1
            if self.active is None or self.pending_count >= self.confirm_samples:
                self.active = raw
                self.pending = None
                self.pending_count = 0

        confidence = 0.0
        if self.active is not None:
            if raw == self.active and distance is not None:
                confidence = max(0.05, 1.0 - distance / self.control_radius)
            else:
                confidence = 0.35
        return PossessionSample(
            frame=frame.frame,
            time_sec=frame.time_sec,
            team=self.active[0] if self.active else None,
            track_id=self.active[1] if self.active else None,
            distance_m=distance if raw == self.active else None,
            confidence=round(confidence, 4),
        )


def nearest_ball_player(
    frame: FrameState,
    radius: float,
) -> tuple[Observation | None, float | None]:
    ball = frame.ball
    if ball is None or not ball.has_pitch_position:
        return None, None
    best: Observation | None = None
    best_distance = float("inf")
    seen_tracks: set[tuple[str | None, int]] = set()
    for player in frame.players:
        key = (player.team, player.track_id)
        if key in seen_tracks or not player.has_pitch_position:
            continue
        seen_tracks.add(key)
        distance = math.hypot(
            float(player.pitch_x) - float(ball.pitch_x),
            float(player.pitch_y) - float(ball.pitch_y),
        )
        if distance < best_distance:
            best, best_distance = player, distance
    if best is None or best_distance > radius:
        return None, None
    return best, best_distance


def attack_direction(frame: FrameState, team: str) -> int:
    goalkeepers = [
        player
        for player in frame.players
        if player.team == team and player.role == "goalkeeper" and player.pitch_x is not None
    ]
    if goalkeepers:
        mean_x = sum(float(player.pitch_x) for player in goalkeepers) / len(goalkeepers)
        return 1 if mean_x < 0 else -1
    return 1 if team == "left" else -1


def ball_point(frame: FrameState) -> tuple[float, float] | None:
    ball = frame.ball
    if ball is None or not ball.has_pitch_position:
        return None
    return float(ball.pitch_x), float(ball.pitch_y)


def shot_geometry(x: float, y: float, goal_x: float) -> tuple[float, float, float]:
    goal_half_width = 3.66
    upper = (goal_x - x, goal_half_width - y)
    lower = (goal_x - x, -goal_half_width - y)
    denom = max(math.hypot(*upper) * math.hypot(*lower), 1e-6)
    cosine = max(-1.0, min(1.0, (upper[0] * lower[0] + upper[1] * lower[1]) / denom))
    angle = math.acos(cosine)
    distance = math.hypot(goal_x - x, y)
    in_box = abs(x) >= HALF_LENGTH - 16.5 and abs(y) <= 20.16
    xg_lite = 0.02 + 0.35 * min(1.0, angle / math.radians(55))
    xg_lite += 0.28 * math.exp(-distance / 18.0) + (0.12 if in_box else 0.0)
    return angle, distance, max(0.01, min(0.75, xg_lite))


def shot_context(
    frame: FrameState,
    attacking_team: str,
    shot_x: float,
    shot_y: float,
    goal_x: float,
    xg_lite: float,
) -> dict[str, float | int | bool | None]:
    """Add interpretable pressure and goalkeeper features to geometry-only xG.

    The resulting score remains an uncalibrated proxy. Its purpose is to expose
    model-ready visual features and improve ranking, not imitate an official xG feed.
    """

    opponent = "right" if attacking_team == "left" else "left"
    defenders = [
        player
        for player in frame.players
        if player.team == opponent and player.has_pitch_position and player.role != "goalkeeper"
    ]
    distances = [
        math.hypot(float(player.pitch_x) - shot_x, float(player.pitch_y) - shot_y)
        for player in defenders
    ]
    nearest_defender = min(distances) if distances else None
    pressure_count = sum(distance <= 5.0 for distance in distances)

    shot_dx = goal_x - shot_x
    shot_dy = -shot_y
    shot_length_sq = max(shot_dx * shot_dx + shot_dy * shot_dy, 1e-6)
    blocking_defenders = 0
    for player in defenders:
        rel_x = float(player.pitch_x) - shot_x
        rel_y = float(player.pitch_y) - shot_y
        projection = (rel_x * shot_dx + rel_y * shot_dy) / shot_length_sq
        if not 0.05 < projection < 0.95:
            continue
        perpendicular = abs(rel_x * shot_dy - rel_y * shot_dx) / math.sqrt(shot_length_sq)
        if perpendicular <= 1.2 + 1.8 * projection:
            blocking_defenders += 1

    goalkeepers = [
        player
        for player in frame.players
        if player.team == opponent and player.role == "goalkeeper" and player.has_pitch_position
    ]
    goalkeeper = min(
        goalkeepers,
        key=lambda player: math.hypot(float(player.pitch_x) - goal_x, float(player.pitch_y)),
        default=None,
    )
    goalkeeper_depth = abs(goal_x - float(goalkeeper.pitch_x)) if goalkeeper else None
    goalkeeper_lateral = abs(float(goalkeeper.pitch_y)) if goalkeeper else None
    goalkeeper_line_distance = None
    if goalkeeper:
        rel_x = float(goalkeeper.pitch_x) - shot_x
        rel_y = float(goalkeeper.pitch_y) - shot_y
        projection = max(0.0, min(1.0, (rel_x * shot_dx + rel_y * shot_dy) / shot_length_sq))
        closest_x = shot_x + projection * shot_dx
        closest_y = shot_y + projection * shot_dy
        goalkeeper_line_distance = math.hypot(float(goalkeeper.pitch_x) - closest_x, float(goalkeeper.pitch_y) - closest_y)

    base = max(0.01, min(0.99, xg_lite))
    logit = math.log(base / (1.0 - base))
    if nearest_defender is not None:
        logit -= 0.75 * math.exp(-nearest_defender / 3.0)
    logit -= 0.18 * max(0, pressure_count - 1)
    logit -= 0.22 * blocking_defenders
    if goalkeeper_line_distance is not None:
        logit -= 0.48 * math.exp(-goalkeeper_line_distance / 2.2)
    if goalkeeper_depth is not None:
        logit += 0.08 * min(goalkeeper_depth / 8.0, 1.0)
    contextual_proxy = 1.0 / (1.0 + math.exp(-logit))
    return {
        "nearest_defender_m": round(nearest_defender, 3) if nearest_defender is not None else None,
        "defenders_within_5m": pressure_count,
        "blocking_defenders": blocking_defenders,
        "goalkeeper_observed": goalkeeper is not None,
        "goalkeeper_depth_m": round(goalkeeper_depth, 3) if goalkeeper_depth is not None else None,
        "goalkeeper_lateral_offset_m": round(goalkeeper_lateral, 3) if goalkeeper_lateral is not None else None,
        "goalkeeper_shot_line_distance_m": round(goalkeeper_line_distance, 3) if goalkeeper_line_distance is not None else None,
        "xg_context_proxy": round(max(0.01, min(0.85, contextual_proxy)), 4),
    }


@register_analyzer("event_timeline")
class EventTimelineAnalyzer(TacticalAnalyzer):
    priority = "P0"
    description_zh = "生成触球、接球、持球推进、传球、球权转换、射门和角球候选时间轴"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        radius = float(self.config.get("control_radius_m", 7.5))
        confirm_samples = int(self.config.get("possession_confirm_samples", 3))
        release_samples = int(self.config.get("possession_release_samples", 4))
        min_pass_distance = float(self.config.get("min_pass_distance_m", 4.0))
        min_carry_distance = float(self.config.get("min_carry_distance_m", 6.0))
        min_carry_progress = float(self.config.get("min_carry_progress_m", 4.0))
        min_carry_duration = float(self.config.get("min_carry_duration_sec", 0.7))
        max_carry_duration = float(self.config.get("max_carry_duration_sec", 12.0))
        shot_speed = float(self.config.get("shot_speed_mps", 18.0))
        max_ball_speed = float(self.config.get("max_ball_speed_mps", 55.0))
        shot_owner_lookback_sec = float(self.config.get("shot_owner_lookback_sec", 0.8))
        max_findings = int(self.config.get("max_event_findings", 16))
        decoder = ConstrainedPossessionDecoder(radius, confirm_samples, release_samples)

        events: list[TimelineEvent] = []
        path: list[PossessionSample] = []
        previous_owner: PossessionSample | None = None
        owner_start_ball: tuple[float, float] | None = None
        owner_last_ball: tuple[float, float] | None = None
        owner_start_time: float | None = None
        owner_start_frame: int | None = None
        previous_ball: tuple[int, float, float] | None = None
        last_shot_time = -999.0
        corner_ready: tuple[float, str | None] | None = None
        last_corner_event_time = -999.0
        corner_cooldown_sec = float(self.config.get("corner_cooldown_sec", 8.0))

        for frame in context.frames:
            possession = decoder.update(frame)
            path.append(possession)
            current_ball = ball_point(frame)

            if possession.track_id is not None and (
                previous_owner is None
                or possession.track_id != previous_owner.track_id
                or possession.team != previous_owner.team
            ):
                if (
                    previous_owner is not None
                    and owner_start_ball
                    and owner_last_ball
                    and owner_start_time is not None
                ):
                    carry_distance = math.hypot(
                        owner_last_ball[0] - owner_start_ball[0],
                        owner_last_ball[1] - owner_start_ball[1],
                    )
                    carry_duration = frame.time_sec - owner_start_time
                    carry_direction = attack_direction(frame, str(previous_owner.team))
                    carry_progress = (owner_last_ball[0] - owner_start_ball[0]) * carry_direction
                    if (
                        min_carry_duration <= carry_duration <= max_carry_duration
                        and carry_distance >= min_carry_distance
                        and carry_progress >= min_carry_progress
                    ):
                        events.append(
                            TimelineEvent(
                                event_id=f"event-{len(events) + 1:04d}",
                                event_type="carry_candidate",
                                frame=owner_start_frame or frame.frame,
                                time_sec=round(owner_start_time, 3),
                                team=previous_owner.team,
                                actor_track_id=previous_owner.track_id,
                                target_track_id=None,
                                confidence=round(min(0.9, 0.5 + carry_progress / 55.0 + carry_distance / 120.0), 4),
                                label_zh="持球推进候选",
                                metrics={
                                    "start_sec": round(owner_start_time, 3),
                                    "end_sec": round(frame.time_sec, 3),
                                    "duration_sec": round(carry_duration, 3),
                                    "distance_m": round(carry_distance, 3),
                                    "forward_progress_m": round(carry_progress, 3),
                                    "start_x": round(owner_start_ball[0], 3),
                                    "start_y": round(owner_start_ball[1], 3),
                                    "end_x": round(owner_last_ball[0], 3),
                                    "end_y": round(owner_last_ball[1], 3),
                                },
                                event_subtype="controlled_carry",
                                outcome="completed",
                                start_x=round(owner_start_ball[0], 3),
                                start_y=round(owner_start_ball[1], 3),
                                end_x=round(owner_last_ball[0], 3),
                                end_y=round(owner_last_ball[1], 3),
                                evidence_start_sec=max(0.0, round(owner_start_time - 1.0, 3)),
                                evidence_end_sec=round(frame.time_sec + 1.0, 3),
                            )
                        )
                if previous_owner is not None and owner_start_ball and current_ball:
                    distance = math.hypot(
                        current_ball[0] - owner_start_ball[0],
                        current_ball[1] - owner_start_ball[1],
                    )
                    if possession.team == previous_owner.team and distance >= min_pass_distance:
                        event_type = "pass_candidate"
                        label = "传球候选"
                        confidence = min(0.94, 0.48 + distance / 65.0 + possession.confidence * 0.25)
                    elif possession.team != previous_owner.team:
                        event_type = "possession_change"
                        label = "球权转换候选"
                        confidence = min(0.91, 0.50 + possession.confidence * 0.35)
                    else:
                        event_type = "touch_change"
                        label = "同队触球切换"
                        confidence = 0.45
                    if event_type != "touch_change":
                        transfer_event = TimelineEvent(
                            event_id=f"event-{len(events) + 1:04d}",
                            event_type=event_type,
                            frame=frame.frame,
                            time_sec=round(frame.time_sec, 3),
                            team=possession.team,
                            actor_track_id=previous_owner.track_id,
                            target_track_id=possession.track_id,
                            confidence=round(confidence, 4),
                            label_zh=label,
                            metrics={
                                "ball_displacement_m": round(distance, 2),
                                "losing_team": previous_owner.team if event_type == "possession_change" else None,
                                "gaining_team": possession.team,
                                "start_x": round(owner_start_ball[0], 3),
                                "start_y": round(owner_start_ball[1], 3),
                                "end_x": round(current_ball[0], 3),
                                "end_y": round(current_ball[1], 3),
                            },
                            event_subtype="completed_pass" if event_type == "pass_candidate" else "stable_team_change",
                            outcome="success",
                            start_x=round(owner_start_ball[0], 3),
                            start_y=round(owner_start_ball[1], 3),
                            end_x=round(current_ball[0], 3),
                            end_y=round(current_ball[1], 3),
                            evidence_start_sec=max(0.0, round(frame.time_sec - 2.0, 3)),
                            evidence_end_sec=round(frame.time_sec + 2.0, 3),
                        )
                        events.append(transfer_event)
                        if event_type == "pass_candidate":
                            events.append(
                                TimelineEvent(
                                    event_id=f"event-{len(events) + 1:04d}",
                                    event_type="receive_candidate",
                                    frame=frame.frame,
                                    time_sec=round(frame.time_sec, 3),
                                    team=possession.team,
                                    actor_track_id=possession.track_id,
                                    target_track_id=None,
                                    confidence=round(max(0.4, confidence - 0.05), 4),
                                    label_zh="接球候选",
                                    metrics={
                                        "pass_event_id": transfer_event.event_id,
                                        "from_track_id": previous_owner.track_id,
                                        "ball_displacement_m": round(distance, 2),
                                    },
                                    event_subtype="stable_receive",
                                    outcome="success",
                                    end_x=round(current_ball[0], 3),
                                    end_y=round(current_ball[1], 3),
                                    evidence_start_sec=max(0.0, round(frame.time_sec - 1.5, 3)),
                                    evidence_end_sec=round(frame.time_sec + 1.5, 3),
                                    related_event_ids=[transfer_event.event_id],
                                )
                            )
                events.append(
                    TimelineEvent(
                        event_id=f"event-{len(events) + 1:04d}",
                        event_type="touch_candidate",
                        frame=frame.frame,
                        time_sec=round(frame.time_sec, 3),
                        team=possession.team,
                        actor_track_id=possession.track_id,
                        target_track_id=None,
                        confidence=round(max(0.4, possession.confidence), 4),
                        label_zh="稳定触球候选",
                        metrics={"distance_to_ball_m": possession.distance_m},
                        event_subtype="stable_control",
                        outcome="success",
                        end_x=round(current_ball[0], 3) if current_ball else None,
                        end_y=round(current_ball[1], 3) if current_ball else None,
                        evidence_start_sec=max(0.0, round(frame.time_sec - 1.0, 3)),
                        evidence_end_sec=round(frame.time_sec + 1.0, 3),
                    )
                )
                previous_owner = possession
                owner_start_ball = current_ball
                owner_last_ball = current_ball
                owner_start_time = frame.time_sec
                owner_start_frame = frame.frame
            elif possession.track_id is not None:
                previous_owner = possession
                if owner_start_ball is None:
                    owner_start_ball = current_ball
                    owner_start_time = frame.time_sec
                    owner_start_frame = frame.frame
                if current_ball is not None:
                    owner_last_ball = current_ball

            if current_ball and previous_ball:
                prev_frame, prev_x, prev_y = previous_ball
                dt = (frame.frame - prev_frame) / max(context.fps, 1e-6)
                if 0 < dt <= 0.5:
                    vx = (current_ball[0] - prev_x) / dt
                    vy = (current_ball[1] - prev_y) / dt
                    speed = math.hypot(vx, vy)
                    recent_owner = (
                        previous_owner
                        if previous_owner and frame.time_sec - previous_owner.time_sec <= shot_owner_lookback_sec
                        else None
                    )
                    team = possession.team or (recent_owner.team if recent_owner else None)
                    if team and frame.time_sec - last_shot_time >= 1.2:
                        direction = attack_direction(frame, team)
                        goal_x = HALF_LENGTH * direction
                        in_attacking_third = prev_x * direction > HALF_LENGTH / 3.0
                        toward_goal = vx * direction > max(4.0, abs(vy) * 0.35)
                        if shot_speed <= speed <= max_ball_speed and in_attacking_third and toward_goal:
                            angle, distance, xg = shot_geometry(prev_x, prev_y, goal_x)
                            context_features = shot_context(frame, team, prev_x, prev_y, goal_x, xg)
                            confidence = min(0.96, 0.53 + (speed - shot_speed) / 45.0 + xg * 0.25)
                            events.append(
                                TimelineEvent(
                                    event_id=f"event-{len(events) + 1:04d}",
                                    event_type="shot_candidate",
                                    frame=frame.frame,
                                    time_sec=round(frame.time_sec, 3),
                                    team=team,
                                    actor_track_id=possession.track_id or (recent_owner.track_id if recent_owner else None),
                                    target_track_id=None,
                                    confidence=round(confidence, 4),
                                    label_zh="射门候选",
                                    metrics={
                                        "shot_x": round(prev_x, 3),
                                        "shot_y": round(prev_y, 3),
                                        "in_penalty_area": bool(
                                            prev_x * direction >= HALF_LENGTH - 16.5 and abs(prev_y) <= 20.16
                                        ),
                                        "shot_channel": (
                                            "left" if prev_y * direction > HALF_WIDTH / 3.0
                                            else "right" if prev_y * direction < -HALF_WIDTH / 3.0
                                            else "center"
                                        ),
                                        "ball_speed_mps": round(speed, 2),
                                        "shot_distance_m": round(distance, 2),
                                        "shot_angle_deg": round(math.degrees(angle), 2),
                                        "xg_lite": round(xg, 4),
                                        **context_features,
                                    },
                                    event_subtype="shot_unknown_result",
                                    outcome="unknown",
                                    start_x=round(prev_x, 3),
                                    start_y=round(prev_y, 3),
                                    evidence_start_sec=max(0.0, round(frame.time_sec - 3.0, 3)),
                                    evidence_end_sec=round(frame.time_sec + 5.0, 3),
                                )
                            )
                            last_shot_time = frame.time_sec

                    near_corner = (
                        abs(current_ball[0]) >= HALF_LENGTH - 5.5
                        and abs(current_ball[1]) >= HALF_WIDTH - 5.5
                    )
                    if (
                        near_corner
                        and speed < 3.0
                        and frame.time_sec - last_corner_event_time >= corner_cooldown_sec
                    ):
                        corner_ready = (frame.time_sec, team)
                    elif corner_ready and speed > 6.0 and frame.time_sec - corner_ready[0] <= 4.0:
                        events.append(
                            TimelineEvent(
                                event_id=f"event-{len(events) + 1:04d}",
                                event_type="corner_candidate",
                                frame=frame.frame,
                                time_sec=round(frame.time_sec, 3),
                                team=corner_ready[1],
                                actor_track_id=possession.track_id,
                                target_track_id=None,
                                confidence=0.72,
                                label_zh="角球开出候选",
                                metrics={"restart_speed_mps": round(speed, 2)},
                                event_subtype="corner",
                                outcome="unknown",
                                start_x=round(prev_x, 3),
                                start_y=round(prev_y, 3),
                                evidence_start_sec=max(0.0, round(frame.time_sec - 4.0, 3)),
                                evidence_end_sec=round(frame.time_sec + 8.0, 3),
                            )
                        )
                        last_corner_event_time = frame.time_sec
                        corner_ready = None
                    elif corner_ready and frame.time_sec - corner_ready[0] > 4.0:
                        corner_ready = None
            if current_ball:
                previous_ball = (frame.frame, current_ball[0], current_ball[1])

        counts = Counter(event.event_type for event in events)
        controlled = [sample for sample in path if sample.team in {"left", "right"}]
        possession_counts = Counter(sample.team for sample in controlled)
        total_controlled = sum(possession_counts.values())
        summary = {
            "decoder": "constrained_possession_path_v1",
            "events": dict(counts),
            "event_count": len(events),
            "controlled_frame_ratio": round(len(controlled) / max(len(path), 1), 4),
            "possession_proxy": {
                team: round(possession_counts[team] / max(total_controlled, 1), 4)
                for team in ("left", "right")
            },
            "shooting_structure": self._shooting_structure(events),
        }

        ranked_events = sorted(
            [
                event
                for event in events
                if event.event_type not in {"touch_candidate", "receive_candidate"}
            ],
            key=lambda event: (
                event.event_type == "shot_candidate",
                event.event_type == "corner_candidate",
                event.confidence,
            ),
            reverse=True,
        )[:max_findings]
        findings = [self._event_finding(event) for event in ranked_events]
        findings.extend(self._shooting_structure_findings(summary["shooting_structure"], context))
        warnings = [
            "事件为 GSR 轨迹上的候选，不等同于 SoccerNet 人工事件标签。",
            "球权路径使用时序约束和最近球员距离，遮挡时可能保留上一稳定状态。",
        ]
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary=summary,
            findings=findings,
            events=events,
            warnings=warnings,
        )

    @staticmethod
    def _shooting_structure(events: list[TimelineEvent]) -> dict[str, dict[str, object]]:
        structure: dict[str, dict[str, object]] = {}
        for team in ("left", "right"):
            shots = [event for event in events if event.event_type == "shot_candidate" and event.team == team]
            distance_buckets = Counter()
            channels = Counter()
            pressure_buckets = Counter()
            for shot in shots:
                distance = float(shot.metrics.get("shot_distance_m", 0.0))
                distance_buckets["close"] += distance <= 12.0
                distance_buckets["medium"] += 12.0 < distance <= 22.0
                distance_buckets["long"] += distance > 22.0
                channels[str(shot.metrics.get("shot_channel", "unknown"))] += 1
                pressure = int(shot.metrics.get("defenders_within_5m", 0))
                pressure_buckets["low"] += pressure == 0
                pressure_buckets["medium"] += pressure == 1
                pressure_buckets["high"] += pressure >= 2
            structure[team] = {
                "shot_candidates": len(shots),
                "in_penalty_area": sum(bool(shot.metrics.get("in_penalty_area")) for shot in shots),
                "outside_penalty_area": sum(not bool(shot.metrics.get("in_penalty_area")) for shot in shots),
                "distance_buckets": dict(distance_buckets),
                "channel_counts": dict(channels),
                "pressure_buckets": dict(pressure_buckets),
                "mean_distance_m": round(
                    sum(float(shot.metrics.get("shot_distance_m", 0.0)) for shot in shots) / len(shots), 3
                ) if shots else None,
                "mean_angle_deg": round(
                    sum(float(shot.metrics.get("shot_angle_deg", 0.0)) for shot in shots) / len(shots), 3
                ) if shots else None,
                "mean_xg_context_proxy": round(
                    sum(float(shot.metrics.get("xg_context_proxy", shot.metrics.get("xg_lite", 0.0))) for shot in shots)
                    / len(shots),
                    4,
                ) if shots else None,
                "total_xg_context_proxy": round(
                    sum(float(shot.metrics.get("xg_context_proxy", shot.metrics.get("xg_lite", 0.0))) for shot in shots),
                    4,
                ),
            }
        return structure

    def _shooting_structure_findings(
        self,
        structure: dict[str, dict[str, object]],
        context: AnalysisContext,
    ) -> list[Finding]:
        findings: list[Finding] = []
        channel_labels = {"left": "左侧", "center": "中路", "right": "右侧", "unknown": "未知通道"}
        for team in ("left", "right"):
            metrics = structure[team]
            count = int(metrics["shot_candidates"])
            if count == 0:
                continue
            channels = dict(metrics["channel_counts"])
            top_channel = max(channels, key=channels.get) if channels else "unknown"
            team_label = "左队" if team == "left" else "右队"
            findings.append(
                Finding(
                    finding_id=f"shooting-structure-{team}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="shooting_structure",
                    title_zh=f"{team_label}射门结构：以{channel_labels.get(top_channel, top_channel)}为主",
                    summary_zh=(
                        f"共识别{count}次射门候选，其中禁区内{metrics['in_penalty_area']}次、"
                        f"禁区外{metrics['outside_penalty_area']}次，平均距离{float(metrics['mean_distance_m'] or 0):.1f}米。"
                    ),
                    confidence=round(min(0.9, 0.5 + count * 0.05), 4),
                    start_sec=context.frames[0].time_sec if context.frames else None,
                    end_sec=context.duration_sec,
                    metrics=metrics,
                    evidence=[
                        Evidence(
                            frame=None,
                            time_sec=None,
                            description_zh="按射门位置、距离、通道、角度和防守压力聚合。",
                            metrics=metrics,
                        )
                    ],
                    limitations_zh=[
                        "当前没有官方射正、进球结果标签，不能据此计算真实射正率。",
                        "xG 上下文值是未校准视觉代理，仅用于同一视频内候选排序。",
                    ],
                )
            )
        return findings

    def _event_finding(self, event: TimelineEvent) -> Finding:
        team_label = {"left": "左队", "right": "右队"}.get(event.team, "未知球队")
        details = {
            "touch_candidate": "球员进入稳定控球半径，形成触球候选。",
            "receive_candidate": "同队传球后形成新的稳定持球人。",
            "carry_candidate": "同一持球人在连续控制期间完成了明确向前位移。",
            "pass_candidate": "同队稳定持球人发生切换，并伴随足球位移。",
            "possession_change": "稳定持球状态跨队切换，可用于定位攻防转换片段。",
            "shot_candidate": "足球在进攻三区高速朝球门方向运动。",
            "corner_candidate": "足球在角旗区低速停留后被快速开出。",
        }
        return Finding(
            finding_id=f"timeline-{event.event_id}",
            analyzer=self.name,
            priority=self.priority,
            category=event.event_type,
            title_zh=f"{event.time_sec:.2f}秒 {team_label}{event.label_zh}",
            summary_zh=details.get(event.event_type, event.label_zh),
            confidence=event.confidence,
            start_sec=max(0.0, event.time_sec - 3.0),
            end_sec=event.time_sec + 4.0,
            metrics=event.metrics,
            evidence=[
                Evidence(
                    frame=event.frame,
                    time_sec=event.time_sec,
                    description_zh=event.label_zh,
                    metrics={
                        "actor_track_id": event.actor_track_id,
                        "target_track_id": event.target_track_id,
                        **event.metrics,
                    },
                )
            ],
            limitations_zh=[
                "需要回看原视频确认真实触球、越位和比赛中断状态。",
                "xg_context_proxy 是未校准的视觉排序指标，不可替代基于大规模射门结果训练的正式 xG。",
            ] if event.event_type == "shot_candidate" else ["需要回看原视频确认真实触球、越位和比赛中断状态。"],
        )
