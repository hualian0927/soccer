"""Pure-vision phase-of-play segmentation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ..base import TacticalAnalyzer, register_analyzer
from ..features import build_motion_index, motion_for, unique_team_players
from ..models import HALF_LENGTH, AnalysisContext, AnalyzerOutput, Evidence, Finding, FrameState
from .events import ConstrainedPossessionDecoder, attack_direction


PHASE_ZH = {
    "build_up": "后场组织",
    "progression": "中场推进",
    "final_third": "前场进攻",
    "attacking_transition": "进攻转换",
    "long_ball": "长传推进",
    "set_piece": "定位球重启",
}


def _forward_ball_speed(frame: FrameState, team: str, motion_index: dict) -> float:
    ball_motion = motion_for(motion_index, frame.ball)
    if ball_motion is None:
        return 0.0
    return ball_motion.vx * attack_direction(frame, team)


def _phase(frame: FrameState, team: str, seconds_since_gain: float, motion_index: dict) -> str:
    ball = frame.ball
    if ball is None or not ball.has_pitch_position:
        return "unknown"
    progress = float(ball.pitch_x) * attack_direction(frame, team)
    forward_speed = _forward_ball_speed(frame, team, motion_index)
    ball_motion = motion_for(motion_index, ball)
    near_corner = abs(float(ball.pitch_x)) >= HALF_LENGTH - 5.5 and abs(float(ball.pitch_y)) >= 28.0
    if near_corner and (ball_motion is None or ball_motion.speed <= 4.0):
        return "set_piece"
    if ball_motion is not None and ball_motion.speed >= 14.0 and forward_speed >= 6.0:
        return "long_ball"
    if seconds_since_gain <= 5.0 and forward_speed >= 3.0:
        return "attacking_transition"
    if progress < -HALF_LENGTH / 3.0:
        return "build_up"
    if progress > HALF_LENGTH / 3.0:
        return "final_third"
    return "progression"


def _defensive_block(frame: FrameState, team: str) -> str | None:
    players = unique_team_players(frame, team)
    if len(players) < 5:
        return None
    direction = attack_direction(frame, team)
    centroid_progress = sum(float(player.pitch_x) * direction for player in players) / len(players)
    if centroid_progress >= 9.0:
        return "high_block"
    if centroid_progress <= -9.0:
        return "low_block"
    return "mid_block"


def _close_segment(segments: list[dict[str, Any]], active: dict[str, Any] | None, end_time: float) -> None:
    if active is None:
        return
    duration = max(0.0, end_time - active["start_sec"])
    if duration >= 0.6:
        segments.append({**active, "end_sec": round(end_time, 3), "duration_sec": round(duration, 3)})


@register_analyzer("phase_of_play")
class PhaseOfPlayAnalyzer(TacticalAnalyzer):
    priority = "P1"
    description_zh = "按球权、球区和运动方向划分后场组织、推进、前场与转换阶段"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        decoder = ConstrainedPossessionDecoder(
            float(self.config.get("control_radius_m", 7.5)),
            int(self.config.get("possession_confirm_samples", 3)),
            int(self.config.get("possession_release_samples", 4)),
        )
        motion_index = build_motion_index(
            context,
            max_gap_sec=float(self.config.get("max_motion_gap_sec", 0.6)),
            smoothing=float(self.config.get("motion_smoothing", 0.55)),
        )
        segments: list[dict[str, Any]] = []
        active: dict[str, Any] | None = None
        last_team: str | None = None
        gain_time = 0.0
        block_counts: dict[str, Counter] = {"left": Counter(), "right": Counter()}
        state_durations: dict[str, Counter] = {"left": Counter(), "right": Counter()}
        transition_window = float(self.config.get("state_transition_window_sec", 3.0))
        previous_time: float | None = None

        for frame in context.frames:
            possession = decoder.update(frame)
            delta_time = 0.0 if previous_time is None else max(0.0, frame.time_sec - previous_time)
            previous_time = frame.time_sec
            if possession.team not in {"left", "right"}:
                _close_segment(segments, active, frame.time_sec)
                active = None
                continue
            if possession.team != last_team:
                gain_time = frame.time_sec
                last_team = possession.team
            for team in ("left", "right"):
                if possession.team == team:
                    state = "regaining_possession" if frame.time_sec - gain_time <= transition_window else "in_possession"
                else:
                    state = "losing_possession" if frame.time_sec - gain_time <= transition_window else "out_of_possession"
                state_durations[team][state] += delta_time
            phase = _phase(frame, possession.team, frame.time_sec - gain_time, motion_index)
            opponent = "right" if possession.team == "left" else "left"
            block = _defensive_block(frame, opponent)
            if block:
                block_counts[opponent][block] += 1
            key = (possession.team, phase)
            if active is None or (active["team"], active["phase"]) != key:
                _close_segment(segments, active, frame.time_sec)
                active = {
                    "team": possession.team,
                    "phase": phase,
                    "start_sec": round(frame.time_sec, 3),
                    "start_frame": frame.frame,
                    "defending_team": opponent,
                    "defensive_block": block,
                }
        if context.frames:
            _close_segment(segments, active, context.frames[-1].time_sec)

        findings: list[Finding] = []
        by_team: dict[str, dict[str, Any]] = {}
        for team in ("left", "right"):
            team_segments = [segment for segment in segments if segment["team"] == team and segment["phase"] != "unknown"]
            durations = Counter()
            for segment in team_segments:
                durations[segment["phase"]] += segment["duration_sec"]
            by_team[team] = {
                "phase_duration_sec": {name: round(value, 3) for name, value in durations.items()},
                "phase_segments": team_segments,
                "defensive_block_samples": dict(block_counts[team]),
                "four_state_duration_sec": {
                    name: round(value, 3) for name, value in state_durations[team].items()
                },
            }
            if not team_segments:
                continue
            dominant_phase, phase_duration = durations.most_common(1)[0]
            label = "左队" if team == "left" else "右队"
            representative = max(
                (segment for segment in team_segments if segment["phase"] == dominant_phase),
                key=lambda item: item["duration_sec"],
            )
            total = sum(durations.values())
            block = block_counts[team].most_common(1)[0][0] if block_counts[team] else "unknown"
            block_zh = {"high_block": "高位防守", "mid_block": "中位防守", "low_block": "低位防守"}.get(block, "证据不足")
            findings.append(
                Finding(
                    finding_id=f"phase-{team}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="phase_of_play",
                    title_zh=f"{label}主要有球阶段：{PHASE_ZH.get(dominant_phase, dominant_phase)}",
                    summary_zh=f"该阶段累计{phase_duration:.1f}秒，占可判定持球时间{phase_duration / max(total, 1e-6):.0%}；无球站位以{block_zh}样本最多。",
                    confidence=round(min(0.9, 0.5 + total / max(context.duration_sec * 2.0, 1.0)), 4),
                    start_sec=representative["start_sec"],
                    end_sec=representative["end_sec"],
                    metrics=by_team[team],
                    evidence=[Evidence(representative["start_frame"], representative["start_sec"], "最长同类连续阶段。", representative)],
                    limitations_zh=["阶段由视觉球权、场区与运动方向规则生成，转播切镜和比赛中断需要人工复核。"],
                )
            )
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "rule_based_visual_phase_segmentation_v2", "teams": by_team, "segments": segments},
            findings=findings,
            warnings=["阶段分类是可解释规则基线，后续可用 SoccerNet/SkillCorner 标注训练时序模型替换。"],
        )
