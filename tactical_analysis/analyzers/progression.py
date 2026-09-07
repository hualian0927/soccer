"""Pass progression, packing and defensive-line breaking proxies."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from ..base import TacticalAnalyzer, register_analyzer
from ..models import HALF_LENGTH, AnalysisContext, AnalyzerOutput, Evidence, Finding, FrameState
from .events import ConstrainedPossessionDecoder, PossessionSample, attack_direction, ball_point


def _opponent_points(frame: FrameState, team: str) -> dict[int, tuple[float, float]]:
    opponent = "right" if team == "left" else "left"
    return {
        player.track_id: (float(player.pitch_x), float(player.pitch_y))
        for player in frame.players
        if player.team == opponent and player.has_pitch_position
    }


def _packing_and_lines(
    start_frame: FrameState,
    end_frame: FrameState,
    team: str,
    start_ball: tuple[float, float],
    end_ball: tuple[float, float],
) -> tuple[int, int]:
    direction = attack_direction(start_frame, team)
    start_opponents = _opponent_points(start_frame, team)
    end_opponents = _opponent_points(end_frame, team)
    start_progress = start_ball[0] * direction
    end_progress = end_ball[0] * direction
    bypassed = 0
    for track_id in start_opponents.keys() & end_opponents.keys():
        before = start_opponents[track_id][0] * direction
        after = end_opponents[track_id][0] * direction
        if before > start_progress + 1.0 and after < end_progress - 1.0:
            bypassed += 1

    opponent_progress = sorted(point[0] * direction for point in start_opponents.values())
    if len(opponent_progress) < 6 or end_progress <= start_progress:
        return bypassed, 0
    line_centers = [float(np.median(chunk)) for chunk in np.array_split(opponent_progress, 3) if len(chunk) >= 2]
    lines_broken = sum(start_progress + 1.5 < center < end_progress - 1.5 for center in line_centers)
    return bypassed, lines_broken


def extract_pass_records(context: AnalysisContext, config: dict[str, Any]) -> list[dict[str, Any]]:
    decoder = ConstrainedPossessionDecoder(
        float(config.get("control_radius_m", 7.5)),
        int(config.get("possession_confirm_samples", 3)),
        int(config.get("possession_release_samples", 4)),
    )
    minimum_distance = float(config.get("min_pass_distance_m", 4.0))
    previous_owner: PossessionSample | None = None
    last_controlled_ball: tuple[float, float] | None = None
    last_controlled_frame: FrameState | None = None
    records: list[dict[str, Any]] = []

    for frame in context.frames:
        possession = decoder.update(frame)
        current_ball = ball_point(frame)
        changed = (
            possession.track_id is not None
            and previous_owner is not None
            and (possession.track_id != previous_owner.track_id or possession.team != previous_owner.team)
        )
        if changed and possession.team == previous_owner.team and last_controlled_ball and current_ball and last_controlled_frame:
            distance = math.hypot(current_ball[0] - last_controlled_ball[0], current_ball[1] - last_controlled_ball[1])
            if distance >= minimum_distance:
                direction = attack_direction(last_controlled_frame, str(possession.team))
                start_progress = last_controlled_ball[0] * direction
                end_progress = current_ball[0] * direction
                progress = end_progress - start_progress
                lateral = abs(current_ball[1] - last_controlled_ball[1])
                if progress > 3.0:
                    pass_direction = "forward"
                elif progress < -3.0:
                    pass_direction = "backward"
                else:
                    pass_direction = "sideways"
                if distance < 15.0:
                    distance_class = "short"
                elif distance < 30.0:
                    distance_class = "medium"
                else:
                    distance_class = "long"
                progressive_threshold = max(8.0, (HALF_LENGTH - start_progress) * 0.15)
                final_third_entry = start_progress <= HALF_LENGTH / 3.0 < end_progress
                penalty_entry = (
                    (start_progress < HALF_LENGTH - 16.5 or abs(last_controlled_ball[1]) > 20.16)
                    and end_progress >= HALF_LENGTH - 16.5
                    and abs(current_ball[1]) <= 20.16
                )
                start_half_space = 7.0 <= abs(last_controlled_ball[1]) <= 18.0
                end_half_space = 7.0 <= abs(current_ball[1]) <= 18.0
                half_space_entry = (
                    not start_half_space
                    and end_half_space
                    and end_progress >= HALF_LENGTH / 3.0
                )
                bypassed, lines_broken = _packing_and_lines(
                    last_controlled_frame,
                    frame,
                    str(possession.team),
                    last_controlled_ball,
                    current_ball,
                )
                cross_candidate = bool(
                    abs(last_controlled_ball[1]) >= 18.0
                    and abs(current_ball[1]) <= 20.16
                    and end_progress >= HALF_LENGTH - 16.5
                    and lateral >= 8.0
                )
                through_ball_candidate = bool(lines_broken > 0 and progress >= 10.0)
                if through_ball_candidate:
                    pass_type = "through_ball"
                elif cross_candidate:
                    pass_type = "cross"
                elif distance_class == "long":
                    pass_type = "long_pass"
                elif distance_class == "short":
                    pass_type = "short_pass"
                else:
                    pass_type = "medium_pass"
                records.append(
                    {
                        "frame": frame.frame,
                        "time_sec": round(frame.time_sec, 3),
                        "team": possession.team,
                        "actor_track_id": previous_owner.track_id,
                        "target_track_id": possession.track_id,
                        "start_frame": last_controlled_frame.frame,
                        "start_x": round(last_controlled_ball[0], 3),
                        "start_y": round(last_controlled_ball[1], 3),
                        "end_x": round(current_ball[0], 3),
                        "end_y": round(current_ball[1], 3),
                        "distance_m": round(distance, 3),
                        "forward_progress_m": round(progress, 3),
                        "lateral_distance_m": round(lateral, 3),
                        "direction": pass_direction,
                        "distance_class": distance_class,
                        "pass_type": pass_type,
                        "progressive": progress >= progressive_threshold,
                        "final_third_entry": final_third_entry,
                        "penalty_area_entry": penalty_entry,
                        "half_space_entry": half_space_entry,
                        "switch_play": bool(
                            last_controlled_ball[1] * current_ball[1] < 0 and lateral >= 32.0
                        ),
                        "cross_candidate": cross_candidate,
                        "opponents_bypassed": bypassed,
                        "lines_broken": lines_broken,
                        "through_ball_candidate": through_ball_candidate,
                    }
                )
        if possession.track_id is not None:
            if changed:
                last_controlled_ball = current_ball
                last_controlled_frame = frame
            previous_owner = possession
            if possession.distance_m is not None and current_ball:
                last_controlled_ball = current_ball
                last_controlled_frame = frame
    return records


@register_analyzer("progression_analysis")
class ProgressionAnalyzer(TacticalAnalyzer):
    priority = "P2"
    description_zh = "分析传球方向、推进距离、Packing 与穿线候选"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        records = extract_pass_records(context, self.config)
        findings: list[Finding] = []
        candidates = [record for record in records if record["progressive"] or record["lines_broken"] > 0]
        candidates.sort(
            key=lambda item: (item["lines_broken"], item["opponents_bypassed"], item["forward_progress_m"]),
            reverse=True,
        )
        for index, record in enumerate(candidates[: int(self.config.get("max_findings", 8))], 1):
            team_label = "左队" if record["team"] == "left" else "右队"
            category = "line_breaking_pass_candidate" if record["lines_broken"] else "progressive_pass_candidate"
            action = "穿线传球候选" if record["lines_broken"] else "推进传球候选"
            confidence = min(
                0.9,
                0.5 + min(0.18, record["forward_progress_m"] / 100.0) + 0.06 * record["opponents_bypassed"] + 0.08 * record["lines_broken"],
            )
            findings.append(
                Finding(
                    finding_id=f"progression-{index:03d}",
                    analyzer=self.name,
                    priority=self.priority,
                    category=category,
                    title_zh=f"{record['time_sec']:.2f}秒 {team_label}{action}",
                    summary_zh=(
                        f"向前推进{record['forward_progress_m']:.1f}米，绕过{record['opponents_bypassed']}名对手，"
                        f"跨越{record['lines_broken']}条防守线代理。"
                    ),
                    confidence=round(confidence, 4),
                    start_sec=max(0.0, record["time_sec"] - 3.0),
                    end_sec=record["time_sec"] + 3.0,
                    metrics=record,
                    evidence=[Evidence(record["frame"], record["time_sec"], "接球人切换与传球前后空间关系。", record)],
                    limitations_zh=["Packing 与穿线基于当前可见对手及三线几何代理，不是人工事件真值。"],
                )
            )

        by_team: dict[str, dict[str, Any]] = {}
        for team in ("left", "right"):
            team_records = [record for record in records if record["team"] == team]
            directions = Counter(record["direction"] for record in team_records)
            distances = Counter(record["distance_class"] for record in team_records)
            pass_types = Counter(record["pass_type"] for record in team_records)
            by_team[team] = {
                "pass_candidates": len(team_records),
                "direction_counts": dict(directions),
                "distance_counts": dict(distances),
                "pass_type_counts": dict(pass_types),
                "progressive_passes": sum(record["progressive"] for record in team_records),
                "line_breaking_candidates": sum(record["lines_broken"] > 0 for record in team_records),
                "final_third_entries": sum(record["final_third_entry"] for record in team_records),
                "penalty_area_entries": sum(record["penalty_area_entry"] for record in team_records),
                "half_space_entries": sum(record["half_space_entry"] for record in team_records),
                "switch_play_candidates": sum(record["switch_play"] for record in team_records),
                "cross_candidates": sum(record["cross_candidate"] for record in team_records),
                "through_ball_candidates": sum(record["through_ball_candidate"] for record in team_records),
                "total_opponents_bypassed": sum(record["opponents_bypassed"] for record in team_records),
            }
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "visual_pass_progression_and_packing_proxy_v1", "teams": by_team, "passes": records},
            findings=findings,
            warnings=["传球成功以同队稳定持球人切换为代理，足球漏检会影响距离与穿线判断。"],
        )
