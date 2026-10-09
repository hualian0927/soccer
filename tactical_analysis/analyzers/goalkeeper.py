"""Goalkeeper events grounded in sustained ball-to-goalkeeper proximity."""

from __future__ import annotations

import math
from collections import defaultdict

from ..base import TacticalAnalyzer, register_analyzer
from ..models import HALF_LENGTH, AnalysisContext, AnalyzerOutput, Evidence, Finding, TimelineEvent
from .events import ConstrainedPossessionDecoder, EventTimelineAnalyzer, ball_point


def image_ball_distance(keeper, ball) -> float | None:
    """Distance from ball center to the keeper box, in keeper-height units."""
    if not keeper.bbox or not ball or not ball.bbox:
        return None
    box, target = keeper.bbox, ball.bbox
    if any(key not in value or not math.isfinite(value[key]) for value in (box,target) for key in ("x","y","w","h")):
        return float("inf")
    if box.get("h", 0) <= 0 or box.get("w", 0) <= 0:
        return None
    if target.get("h", 0) > box["h"] * 0.45 or target.get("w", 0) > box["w"] * 0.8:
        return float("inf")
    bx, by = target["x"] + target["w"] / 2, target["y"] + target["h"] / 2
    dx = max(box["x"] - bx, 0, bx - box["x"] - box["w"])
    dy = max(box["y"] - by, 0, by - box["y"] - box["h"])
    return math.hypot(dx, dy) / box["h"]


@register_analyzer("goalkeeper_interventions")
class GoalkeeperInterventionAnalyzer(TacticalAnalyzer):
    priority = "P1"
    description_zh = "以二维球场上的持续球门将接近为前提，识别门将接近、控制和扑救干预候选"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        event_output = EventTimelineAnalyzer(self.config).analyze(context)
        shots = [event for event in event_output.events if event.event_type == "shot_candidate"]
        intervention_radius = float(self.config.get("intervention_radius_m", 5.5))
        approach_radius = max(intervention_radius, float(self.config.get("approach_radius_m", 14.0)))
        minimum_duration = float(self.config.get("minimum_proximity_duration_sec", 0.4))
        minimum_samples = int(self.config.get("minimum_proximity_samples", 6))
        minimum_approach = float(self.config.get("minimum_approach_distance_m", 4.5))
        maximum_gap = float(self.config.get("maximum_ball_gap_sec", 0.7))
        cooldown = float(self.config.get("candidate_cooldown_sec", 4.0))
        goal_depth = float(self.config.get("goal_area_depth_m", 14.0))
        goal_half_width = float(self.config.get("goal_area_half_width_m", 20.16))
        maximum_response_speed = float(self.config.get("maximum_goalkeeper_response_mps", 9.0))

        decoder = ConstrainedPossessionDecoder(
            float(self.config.get("control_radius_m", 7.5)),
            int(self.config.get("possession_confirm_samples", 3)),
            int(self.config.get("possession_release_samples", 4)),
        )
        possession_by_frame = {frame.frame: decoder.update(frame) for frame in context.frames}

        observations_by_track: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
        for frame in context.frames:
            ball = ball_point(frame)
            if ball is None or not all(math.isfinite(v) for v in ball):
                continue
            for keeper in frame.players:
                if keeper.role != "goalkeeper" or keeper.team not in {"left", "right"} or not keeper.has_pitch_position:
                    continue
                keeper_x = float(keeper.pitch_x)
                keeper_y = float(keeper.pitch_y)
                if not math.isfinite(keeper_x) or not math.isfinite(keeper_y):
                    continue
                nearest_goal_x = HALF_LENGTH if keeper_x >= 0.0 else -HALF_LENGTH
                depth_from_goal = abs(nearest_goal_x - keeper_x)
                if depth_from_goal > goal_depth or abs(keeper_y) > goal_half_width:
                    continue
                distance = math.hypot(keeper_x - ball[0], keeper_y - ball[1])
                if distance > approach_radius:
                    continue
                image_distance = image_ball_distance(keeper, frame.ball)
                if distance <= intervention_radius and image_distance is not None and image_distance > 0.65:
                    continue
                possession = possession_by_frame.get(frame.frame)
                observations_by_track[(keeper.team, keeper.track_id)].append(
                    {
                        "frame": frame.frame,
                        "time_sec": frame.time_sec,
                        "track_id": keeper.track_id,
                        "team": keeper.team,
                        "x": keeper_x,
                        "y": keeper_y,
                        "goal_x": nearest_goal_x,
                        "goal_depth_m": depth_from_goal,
                        "ball_x": ball[0],
                        "ball_y": ball[1],
                        "ball_distance_m": distance,
                        "image_distance_heights": image_distance,
                        "controlled": bool(possession and possession.track_id == keeper.track_id),
                    }
                )

        episodes: list[list[dict[str, object]]] = []
        for track_observations in observations_by_track.values():
            current: list[dict[str, object]] = []
            for observation in track_observations:
                gap = float(observation["time_sec"]) - float(current[-1]["time_sec"]) if current else 0
                jump = math.hypot(float(observation["x"]) - float(current[-1]["x"]),
                                  float(observation["y"]) - float(current[-1]["y"])) if current else 0
                if current and (gap > maximum_gap or jump > maximum_response_speed * max(gap, 0.03) + 1.5
                                or float(observation["time_sec"]) - float(current[0]["time_sec"]) > 5.0):
                    episodes.append(current)
                    current = []
                current.append(observation)
            if current:
                episodes.append(current)

        interventions: list[dict[str, object]] = []
        events: list[TimelineEvent] = []
        findings: list[Finding] = []
        last_event_by_team = {"left": -999.0, "right": -999.0}

        for episode in sorted(episodes, key=lambda item: float(item[0]["time_sec"])):
            first = episode[0]
            action = min(episode, key=lambda item: float(item["ball_distance_m"]))
            team = str(action["team"])
            start_time = float(first["time_sec"])
            action_time = float(action["time_sec"])
            duration = float(episode[-1]["time_sec"]) - start_time
            close_samples = sum(float(item["ball_distance_m"]) <= intervention_radius for item in episode)
            control_samples = sum(bool(item["controlled"]) for item in episode)
            minimum_distance = float(action["ball_distance_m"])
            approach_drop = float(first["ball_distance_m"]) - minimum_distance

            sustained = duration >= minimum_duration and close_samples >= minimum_samples
            has_action_evidence = approach_drop >= minimum_approach or control_samples >= 3
            if not sustained or not has_action_evidence or action_time - last_event_by_team[team] < cooldown:
                continue

            response_distance = math.hypot(
                float(action["x"]) - float(first["x"]),
                float(action["y"]) - float(first["y"]),
            )
            response_speed = response_distance / max(action_time-start_time, 0.1)
            if response_speed > maximum_response_speed:
                continue

            related_shot = min(
                (
                    shot for shot in shots
                    if shot.team and shot.team != team and start_time - 2.0 <= shot.time_sec <= action_time
                ),
                key=lambda shot: abs(action_time - shot.time_sec),
                default=None,
            )
            lateral_displacement = abs(float(action["y"]) - float(first["y"]))
            outfield_displacement = float(action["goal_depth_m"]) - float(first["goal_depth_m"])
            if related_shot:
                action_type = "shot_intervention"
                action_label = "门将射门干预待复核"
            elif control_samples >= 3:
                action_type = "ball_control"
                action_label = "门将控球候选"
            else:
                action_type = "ball_approach"
                action_label = "门将接近球路候选"

            metrics = {
                "shot_event_id": related_shot.event_id if related_shot else None,
                "shot_time_sec": related_shot.time_sec if related_shot else None,
                "defending_team": team,
                "goalkeeper_track_id": int(action["track_id"]),
                "goalkeeper_source": "role_label_ball_proximity",
                "action_type": action_type,
                "action_time_sec": round(action_time, 3),
                "minimum_ball_distance_m": round(minimum_distance, 3),
                "approach_start_distance_m": round(float(first["ball_distance_m"]), 3),
                "approach_distance_drop_m": round(approach_drop, 3),
                "proximity_duration_sec": round(duration, 3),
                "proximity_samples": close_samples,
                "control_samples": control_samples,
                "image_distance_heights": action.get("image_distance_heights"),
                "requires_visual_action_review": True,
                "goal_depth_m": round(float(action["goal_depth_m"]), 3),
                "lateral_displacement_m": round(lateral_displacement, 3),
                "outfield_displacement_m": round(outfield_displacement, 3),
                "response_distance_m": round(response_distance, 3),
                "response_speed_mps": round(response_speed, 3),
                "goalkeeper_controlled_after_shot": bool(related_shot and control_samples >= 3),
            }
            confidence = 0.54
            confidence += min(0.18, 0.18 * approach_drop / max(minimum_approach, 1e-6))
            confidence += 0.1 if control_samples >= 3 else 0.0
            confidence += 0.08 if related_shot else 0.0
            confidence += min(0.06, close_samples / max(minimum_samples, 1) * 0.03)
            confidence = min(0.94, confidence)

            event = TimelineEvent(
                event_id=f"goalkeeper-{len(events) + 1:03d}",
                event_type="goalkeeper_intervention_candidate",
                frame=int(action["frame"]),
                time_sec=round(action_time, 3),
                team=team,
                actor_track_id=int(action["track_id"]),
                target_track_id=related_shot.actor_track_id if related_shot else None,
                confidence=round(confidence, 4),
                label_zh=action_label,
                metrics=metrics,
                event_subtype=action_type,
                outcome="controlled" if control_samples >= 3 else "unknown",
                start_x=float(first["ball_x"]),
                start_y=float(first["ball_y"]),
                end_x=float(action["ball_x"]),
                end_y=float(action["ball_y"]),
                evidence_start_sec=max(0.0, action_time - 5.0),
                evidence_end_sec=min(context.duration_sec, action_time + 5.0),
                source="vision_ball_goalkeeper_proximity",
                related_event_ids=[related_shot.event_id] if related_shot else [],
            )
            events.append(event)
            interventions.append(metrics)
            last_event_by_team[team] = action_time
            findings.append(
                Finding(
                    finding_id=f"goalkeeper-intervention-{len(findings) + 1:03d}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="goalkeeper_save_candidate" if related_shot else "goalkeeper_contact_candidate",
                    title_zh=f"{action_time:.2f}秒 {'左队' if team == 'left' else '右队'}{action_label}",
                    summary_zh=(
                        f"球门将距离由{float(first['ball_distance_m']):.1f}米降至{minimum_distance:.1f}米，"
                        f"持续{duration:.1f}秒，{'形成稳定控制' if control_samples >= 3 else '控制结果待确认'}。"
                    ),
                    confidence=event.confidence,
                    start_sec=event.evidence_start_sec,
                    end_sec=event.evidence_end_sec,
                    metrics=metrics,
                    evidence=[Evidence(event.frame, event.time_sec, "二维球场中足球持续进入门将作用范围。", metrics)],
                    limitations_zh=["二维距离不能单独证明扑救类型，是否扑救及动作规范性仍需视频或官方事件复核。"],
                )
            )

        # Image-space contacts retain review candidates when projection is unavailable.
        image_contacts = defaultdict(list)
        for frame in context.frames:
            for keeper in frame.players:
                if keeper.role != "goalkeeper":
                    continue
                image_distance = image_ball_distance(keeper, frame.ball)
                if image_distance is None or image_distance > 0.3:
                    continue
                point = ball_point(frame)
                if point is not None and keeper.has_pitch_position:
                    continue
                image_contacts[(keeper.team, keeper.track_id)].append((frame, image_distance))
        for (team, track_id), samples in image_contacts.items():
            groups = []
            for item in samples:
                if not groups or item[0].time_sec - groups[-1][-1][0].time_sec > 0.3:
                    groups.append([])
                groups[-1].append(item)
            for group in groups:
                if len(group) < 4 or group[-1][0].time_sec - group[0][0].time_sec < 0.12:
                    continue
                frame, distance = min(group, key=lambda item: item[1])
                if any(event.team == team and abs(event.time_sec - frame.time_sec) < cooldown for event in events):
                    continue
                events.append(TimelineEvent(
                    event_id=f"goalkeeper-image-{len(events) + 1:03d}", event_type="goalkeeper_intervention_candidate",
                    frame=frame.frame, time_sec=frame.time_sec, team=team, actor_track_id=track_id,
                    target_track_id=None, confidence=0.5, label_zh="门将附近触球待复核",
                    event_subtype="image_contact", source="keeper_kit_image_proximity",
                    evidence_start_sec=max(0, frame.time_sec - 5),
                    evidence_end_sec=min(context.duration_sec, frame.time_sec + 5),
                    metrics={"action_type": "image_contact", "image_distance_heights": distance,
                             "geometry_available": False, "requires_visual_action_review": True},
                ))
        events.sort(key=lambda event: event.time_sec)
        team_summary = {}
        for team in ("left", "right"):
            faced = [shot for shot in shots if shot.team and shot.team != team]
            items = [item for item in interventions if item["defending_team"] == team]
            team_summary[team] = {
                "shots_faced_candidates": len(faced),
                "intervention_candidates": sum(event.team == team for event in events),
                "controlled_candidates": sum(int(item["control_samples"]) >= 3 for item in items),
                "action_type_counts": {
                    action: sum(item["action_type"] == action for item in items)
                    for action in ("lateral_save", "shot_intervention", "ball_control", "ball_approach")
                },
            }
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "keeper_identity_pitch_image_contact_v3", "teams": team_summary, "interventions": interventions,
                     "keeper_observations": sum(player.role == "goalkeeper" for frame in context.frames for player in frame.players)},
            findings=findings,
            events=events,
            warnings=[
                "门将事件必须满足门区位置、持续球门将接近以及明显趋近或稳定控制。",
                "扑救、射正和成功率没有官方结果标签时只输出候选，不计算真实扑救成功率。",
            ],
        )
