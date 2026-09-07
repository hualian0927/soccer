"""Attacking and defensive transition metrics from visual trajectories."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ..base import TacticalAnalyzer, register_analyzer
from ..features import build_motion_index, motion_for, unique_team_players
from ..models import HALF_LENGTH, AnalysisContext, AnalyzerOutput, Evidence, Finding, TimelineEvent
from .events import ConstrainedPossessionDecoder, attack_direction, ball_point


def _shape_signature(frame, team: str, minimum_players: int = 7) -> tuple[float, float] | None:
    players = {
        player.track_id: player
        for player in frame.players
        if player.team == team and player.has_pitch_position
    }
    if len(players) < minimum_players:
        return None
    points = np.asarray(
        [(float(player.pitch_x), float(player.pitch_y)) for player in players.values()],
        dtype=np.float32,
    )
    return float(np.ptp(points[:, 1])), float(np.ptp(points[:, 0]))


@register_analyzer("transition_analysis")
class TransitionAnalyzer(TacticalAnalyzer):
    priority = "P3"
    description_zh = "分析夺回后的向前行动、推进距离、参攻人数与5秒反抢结果"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        decoder = ConstrainedPossessionDecoder(
            float(self.config.get("control_radius_m", 7.5)),
            int(self.config.get("possession_confirm_samples", 3)),
            int(self.config.get("possession_release_samples", 4)),
        )
        window_sec = float(self.config.get("window_sec", 5.0))
        motion_index = build_motion_index(context)
        path = [(frame, decoder.update(frame)) for frame in context.frames]
        minimum_shape_players = int(self.config.get("minimum_shape_players", 7))
        shape_tolerance = float(self.config.get("shape_recovery_tolerance", 0.22))
        recovery_window_sec = float(self.config.get("shape_recovery_window_sec", 12.0))
        minimum_recovery_sec = float(self.config.get("minimum_shape_recovery_sec", 1.5))
        defensive_references: dict[str, tuple[float, float] | None] = {}
        for team in ("left", "right"):
            samples = [
                signature
                for frame, possession in path
                if possession.team in {"left", "right"} and possession.team != team
                for signature in [_shape_signature(frame, team, minimum_shape_players)]
                if signature is not None
            ]
            defensive_references[team] = (
                (float(np.median([item[0] for item in samples])), float(np.median([item[1] for item in samples])))
                if samples
                else None
            )
        raw_segments: list[dict[str, Any]] = []
        active_segment: dict[str, Any] | None = None
        for index, (_, possession) in enumerate(path):
            if possession.team not in {"left", "right"}:
                continue
            if active_segment is None or active_segment["team"] != possession.team:
                if active_segment is not None:
                    raw_segments.append(active_segment)
                active_segment = {
                    "team": possession.team,
                    "start_index": index,
                    "end_index": index,
                    "start_sec": possession.time_sec,
                    "end_sec": possession.time_sec,
                }
            else:
                active_segment["end_index"] = index
                active_segment["end_sec"] = possession.time_sec
        if active_segment is not None:
            raw_segments.append(active_segment)

        minimum_hold = float(self.config.get("minimum_possession_hold_sec", 0.8))
        stable_segments = [
            segment
            for segment in raw_segments
            if segment["end_sec"] - segment["start_sec"] >= minimum_hold
        ]
        merged_segments: list[dict[str, Any]] = []
        for segment in stable_segments:
            if merged_segments and merged_segments[-1]["team"] == segment["team"]:
                merged_segments[-1]["end_index"] = segment["end_index"]
                merged_segments[-1]["end_sec"] = segment["end_sec"]
            else:
                merged_segments.append(dict(segment))
        changes: list[tuple[int, str, str]] = []
        maximum_gap = float(self.config.get("maximum_unknown_gap_sec", 1.2))
        for previous, current in zip(merged_segments, merged_segments[1:]):
            if current["start_sec"] - previous["end_sec"] <= maximum_gap:
                changes.append((int(current["start_index"]), str(previous["team"]), str(current["team"])))

        records: list[dict[str, Any]] = []
        for index, losing_team, gaining_team in changes:
            start_frame, start_possession = path[index]
            start_ball = ball_point(start_frame)
            if start_ball is None:
                continue
            direction = attack_direction(start_frame, gaining_team)
            start_progress = start_ball[0] * direction
            max_progress = start_progress
            first_forward_action = None
            final_third_time = None
            peak_ball_speed = 0.0
            max_joining = 0
            max_recovery_runners = 0
            regained_by_loser = False
            shape_recovery_sec = None
            recovery_streak = 0
            defensive_reference = defensive_references.get(losing_team)
            for window_frame, window_possession in path[index:]:
                elapsed = window_frame.time_sec - start_frame.time_sec
                if elapsed > max(window_sec, recovery_window_sec):
                    break
                if elapsed <= window_sec:
                    point = ball_point(window_frame)
                    if point:
                        progress = point[0] * direction
                        max_progress = max(max_progress, progress)
                        if first_forward_action is None and progress - start_progress >= 3.0:
                            first_forward_action = elapsed
                        if final_third_time is None and progress >= HALF_LENGTH / 3.0:
                            final_third_time = elapsed
                        ball_motion = motion_for(motion_index, window_frame.ball)
                        if ball_motion:
                            peak_ball_speed = max(peak_ball_speed, ball_motion.speed)
                    joining = 0
                    for player in unique_team_players(window_frame, gaining_team):
                        if player.role == "goalkeeper":
                            continue
                        player_motion = motion_for(motion_index, player)
                        if player_motion and player_motion.speed >= 2.0 and player_motion.vx * direction >= 1.2:
                            joining += 1
                    max_joining = max(max_joining, min(joining, 10))
                    losing_direction = attack_direction(window_frame, losing_team)
                    recovering = 0
                    for player in unique_team_players(window_frame, losing_team):
                        if player.role == "goalkeeper":
                            continue
                        player_motion = motion_for(motion_index, player)
                        if player_motion and player_motion.speed >= 2.0 and player_motion.vx * losing_direction <= -1.2:
                            recovering += 1
                    max_recovery_runners = max(max_recovery_runners, min(recovering, 10))
                    if elapsed >= 0.6 and window_possession.team == losing_team:
                        regained_by_loser = True
                if defensive_reference is not None and elapsed >= minimum_recovery_sec and shape_recovery_sec is None:
                    signature = _shape_signature(window_frame, losing_team, minimum_shape_players)
                    if signature is not None:
                        width_ok = abs(signature[0] - defensive_reference[0]) <= max(5.0, defensive_reference[0] * shape_tolerance)
                        depth_ok = abs(signature[1] - defensive_reference[1]) <= max(5.0, defensive_reference[1] * shape_tolerance)
                        recovery_streak = recovery_streak + 1 if width_ok and depth_ok else 0
                        if recovery_streak >= 3:
                            shape_recovery_sec = max(0.0, elapsed - 2.0 / max(context.fps, 1.0))

            forward_distance = max(0.0, max_progress - start_progress)
            transition_speed = forward_distance / max(window_sec, 1e-6)
            transition_quality = min(
                1.0,
                forward_distance / 45.0 * 0.45
                + int(final_third_time is not None) * 0.2
                + min(max_joining, 6) / 6.0 * 0.2
                + (0.15 if first_forward_action is not None and first_forward_action <= 2.0 else 0.0),
            )
            records.append(
                {
                    "frame": start_frame.frame,
                    "time_sec": round(start_frame.time_sec, 3),
                    "losing_team": losing_team,
                    "gaining_team": gaining_team,
                    "forward_distance_5s_m": round(forward_distance, 3),
                    "transition_speed_mps": round(transition_speed, 3),
                    "attacking_transition_quality_proxy": round(transition_quality, 4),
                    "first_forward_action_sec": round(first_forward_action, 3) if first_forward_action is not None else None,
                    "time_to_final_third_sec": round(final_third_time, 3) if final_third_time is not None else None,
                    "peak_ball_speed_mps": round(peak_ball_speed, 3),
                    "players_joining_attack": max_joining,
                    "recovery_runners": max_recovery_runners,
                    "counterpress_regain_within_5s": regained_by_loser,
                    "defensive_shape_recovery_sec": round(shape_recovery_sec, 3) if shape_recovery_sec is not None else None,
                    "shape_recovered_within_window": shape_recovery_sec is not None,
                    "shape_recovery_window_sec": recovery_window_sec,
                }
            )

        findings: list[Finding] = []
        timeline_events: list[TimelineEvent] = []
        ranked = sorted(records, key=lambda item: (item["forward_distance_5s_m"], item["players_joining_attack"]), reverse=True)
        for number, record in enumerate(ranked[: int(self.config.get("max_findings", 8))], 1):
            team_label = "左队" if record["gaining_team"] == "left" else "右队"
            confidence = min(0.9, 0.52 + record["forward_distance_5s_m"] / 80.0 + record["players_joining_attack"] * 0.025)
            first_action = record["first_forward_action_sec"]
            first_action_text = f"{first_action:.1f}秒" if first_action is not None else "5秒内未确认"
            findings.append(
                Finding(
                    finding_id=f"transition-{number:03d}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="attacking_transition_candidate",
                    title_zh=f"{record['time_sec']:.2f}秒 {team_label}进攻转换候选",
                    summary_zh=(
                        f"5秒内向前推进{record['forward_distance_5s_m']:.1f}米，第一次明确向前行动为{first_action_text}，"
                        f"最多{record['players_joining_attack']}人高速前插。"
                    ),
                    confidence=round(confidence, 4),
                    start_sec=max(0.0, record["time_sec"] - 2.0),
                    end_sec=record["time_sec"] + window_sec,
                    metrics=record,
                    evidence=[Evidence(record["frame"], record["time_sec"], "稳定球权跨队切换后的5秒窗口。", record)],
                    limitations_zh=["推进和参攻人数依赖可见球员，切镜时会低估；球权切换仍需人工复核。"],
                )
            )
            timeline_events.append(
                TimelineEvent(
                    event_id=f"transition-{number:03d}",
                    event_type="transition_speed_candidate",
                    frame=int(record["frame"]),
                    time_sec=float(record["time_sec"]),
                    team=str(record["gaining_team"]),
                    actor_track_id=None,
                    target_track_id=None,
                    confidence=round(confidence, 4),
                    label_zh="攻防转换速度候选",
                    metrics=record,
                )
            )
            losing_label = "左队" if record["losing_team"] == "left" else "右队"
            recovery = record["defensive_shape_recovery_sec"]
            recovery_text = f"约{recovery:.1f}秒恢复接近常态防守宽度/纵深" if recovery is not None else f"{recovery_window_sec:.0f}秒内未确认队形恢复"
            findings.append(
                Finding(
                    finding_id=f"defensive-transition-{number:03d}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="defensive_transition_candidate",
                    title_zh=f"{record['time_sec']:.2f}秒 {losing_label}防守转换候选",
                    summary_zh=(
                        f"最多{record['recovery_runners']}人形成回追代理，{recovery_text}；"
                        f"5秒内反抢夺回：{'是' if record['counterpress_regain_within_5s'] else '否'}。"
                    ),
                    confidence=round(min(0.86, 0.48 + record["recovery_runners"] * 0.04 + (0.1 if recovery is not None else 0.0)), 4),
                    start_sec=max(0.0, record["time_sec"] - 2.0),
                    end_sec=record["time_sec"] + recovery_window_sec,
                    metrics=record,
                    evidence=[Evidence(record["frame"], record["time_sec"], "丢球后的回追和队形恢复窗口。", record)],
                    limitations_zh=["队形恢复以本队全片无球宽度/纵深中位数为参照，切镜会造成缺失或提前匹配。"],
                )
            )

        by_losing_team = {}
        by_gaining_team = {}
        for team in ("left", "right"):
            samples = [record for record in records if record["losing_team"] == team]
            by_losing_team[team] = {
                "losses": len(samples),
                "counterpress_regains_within_5s": sum(record["counterpress_regain_within_5s"] for record in samples),
                "five_second_regain_rate": round(
                    sum(record["counterpress_regain_within_5s"] for record in samples) / max(len(samples), 1),
                    4,
                ),
                "recovery_runner_distribution": dict(Counter(record["recovery_runners"] for record in samples)),
                "shape_recoveries_observed": sum(record["shape_recovered_within_window"] for record in samples),
                "median_shape_recovery_sec": round(
                    float(np.median([record["defensive_shape_recovery_sec"] for record in samples if record["defensive_shape_recovery_sec"] is not None])),
                    3,
                ) if any(record["defensive_shape_recovery_sec"] is not None for record in samples) else None,
            }
            attacking_samples = [record for record in records if record["gaining_team"] == team]
            by_gaining_team[team] = {
                "gains": len(attacking_samples),
                "mean_transition_speed_mps": round(
                    sum(float(record["transition_speed_mps"]) for record in attacking_samples) / len(attacking_samples),
                    3,
                ) if attacking_samples else None,
                "mean_attacking_quality_proxy": round(
                    sum(float(record["attacking_transition_quality_proxy"]) for record in attacking_samples) / len(attacking_samples),
                    4,
                ) if attacking_samples else None,
                "final_third_reaches": sum(record["time_to_final_third_sec"] is not None for record in attacking_samples),
            }
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={
                "method": "visual_five_second_transition_window_v2",
                "teams": by_losing_team,
                "attacking_teams": by_gaining_team,
                "transitions": records,
            },
            findings=findings,
            events=timeline_events,
            warnings=["5秒反抢成功率是短片段视觉代理，需排除界外球、犯规和转播切镜。"],
        )
