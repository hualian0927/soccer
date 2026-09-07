"""Dynamic numerical-superiority episodes around the ball."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ..base import TacticalAnalyzer, register_analyzer
from ..models import HALF_LENGTH, AnalysisContext, AnalyzerOutput, Evidence, Finding, TimelineEvent
from .events import ConstrainedPossessionDecoder, EventTimelineAnalyzer, attack_direction, ball_point


def attack_zone(progress: float) -> str:
    if progress < -HALF_LENGTH / 3.0:
        return "build_up_third"
    if progress > HALF_LENGTH / 3.0:
        return "final_third"
    return "middle_third"


ZONE_LABELS = {
    "build_up_third": "后场",
    "middle_third": "中场",
    "final_third": "进攻三区",
}


@register_analyzer("numerical_superiority")
class NumericalSuperiorityAnalyzer(TacticalAnalyzer):
    priority = "P3"
    description_zh = "识别持续局部多打少，并关联推进、前场进入和射门结果"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        decoder = ConstrainedPossessionDecoder(
            float(self.config.get("control_radius_m", 7.5)),
            int(self.config.get("possession_confirm_samples", 3)),
            int(self.config.get("possession_release_samples", 4)),
        )
        local_radius = float(self.config.get("local_radius_m", 14.0))
        minimum_margin = int(self.config.get("minimum_advantage", 2))
        minimum_attackers = int(self.config.get("minimum_attackers", 3))
        sample_step = max(1, int(self.config.get("sample_step_frames", 5)))
        maximum_gap = float(self.config.get("maximum_episode_gap_sec", 0.65))
        minimum_duration = float(self.config.get("minimum_episode_sec", 0.3))
        outcome_window = float(self.config.get("outcome_window_sec", 8.0))

        path = [(frame, decoder.update(frame)) for frame in context.frames]
        samples: list[dict[str, Any]] = []
        for index, (frame, possession) in enumerate(path):
            if index % sample_step or possession.team not in {"left", "right"}:
                continue
            point = ball_point(frame)
            if point is None:
                continue
            attacking_team = str(possession.team)
            defending_team = "right" if attacking_team == "left" else "left"
            unique_players = {
                (player.team, player.track_id): player
                for player in frame.players
                if player.team in {"left", "right"} and player.has_pitch_position
            }
            local_counts = {"left": 0, "right": 0}
            for player in unique_players.values():
                distance = math.hypot(float(player.pitch_x) - point[0], float(player.pitch_y) - point[1])
                if distance <= local_radius:
                    local_counts[str(player.team)] += 1
            margin = local_counts[attacking_team] - local_counts[defending_team]
            if local_counts[attacking_team] < minimum_attackers or margin < minimum_margin:
                continue
            direction = attack_direction(frame, attacking_team)
            progress = point[0] * direction
            samples.append(
                {
                    "frame": frame.frame,
                    "time_sec": frame.time_sec,
                    "attacking_team": attacking_team,
                    "defending_team": defending_team,
                    "attackers": local_counts[attacking_team],
                    "defenders": local_counts[defending_team],
                    "margin": margin,
                    "ball_x": point[0],
                    "ball_y": point[1],
                    "ball_progress": progress,
                    "zone": attack_zone(progress),
                }
            )

        raw_episodes: list[list[dict[str, Any]]] = []
        active: list[dict[str, Any]] = []
        for sample in samples:
            if active and (
                sample["attacking_team"] != active[-1]["attacking_team"]
                or sample["time_sec"] - active[-1]["time_sec"] > maximum_gap
            ):
                raw_episodes.append(active)
                active = []
            active.append(sample)
        if active:
            raw_episodes.append(active)

        shot_events = [
            event
            for event in EventTimelineAnalyzer(self.config).analyze(context).events
            if event.event_type == "shot_candidate"
        ]
        episodes: list[dict[str, Any]] = []
        for group in raw_episodes:
            duration = group[-1]["time_sec"] - group[0]["time_sec"]
            if duration < minimum_duration:
                continue
            team = group[0]["attacking_team"]
            shot = next(
                (
                    event for event in shot_events
                    if event.team == team and group[0]["time_sec"] <= event.time_sec <= group[-1]["time_sec"] + outcome_window
                ),
                None,
            )
            start_progress = float(group[0]["ball_progress"])
            max_progress = max(float(sample["ball_progress"]) for sample in group)
            entered_final_third = any(float(sample["ball_progress"]) >= HALF_LENGTH / 3.0 for sample in group)
            retained_after = any(
                possession.team == team
                for frame, possession in path
                if group[-1]["time_sec"] + 1.0 <= frame.time_sec <= group[-1]["time_sec"] + 3.0
            )
            mean_margin = sum(int(sample["margin"]) for sample in group) / len(group)
            zone = Counter(str(sample["zone"]) for sample in group).most_common(1)[0][0]
            effectiveness = int(entered_final_third) + int(shot is not None) * 2 + int(retained_after)
            episodes.append(
                {
                    "start_frame": group[0]["frame"],
                    "start_sec": round(float(group[0]["time_sec"]), 3),
                    "end_sec": round(float(group[-1]["time_sec"]), 3),
                    "duration_sec": round(duration, 3),
                    "attacking_team": team,
                    "defending_team": group[0]["defending_team"],
                    "zone": zone,
                    "mean_attackers": round(sum(int(sample["attackers"]) for sample in group) / len(group), 3),
                    "mean_defenders": round(sum(int(sample["defenders"]) for sample in group) / len(group), 3),
                    "mean_advantage": round(mean_margin, 3),
                    "peak_advantage": max(int(sample["margin"]) for sample in group),
                    "forward_progress_m": round(max(0.0, max_progress - start_progress), 3),
                    "entered_final_third": entered_final_third,
                    "shot_within_8s": shot is not None,
                    "shot_time_sec": shot.time_sec if shot else None,
                    "retained_possession_after_episode": retained_after,
                    "effectiveness_score": effectiveness,
                    "sample_count": len(group),
                }
            )

        ranked = sorted(
            episodes,
            key=lambda item: (item["effectiveness_score"], item["peak_advantage"], item["duration_sec"]),
            reverse=True,
        )
        findings: list[Finding] = []
        timeline_events: list[TimelineEvent] = []
        for number, episode in enumerate(ranked[: int(self.config.get("max_findings", 8))], 1):
            team_label = "左队" if episode["attacking_team"] == "left" else "右队"
            outcome_parts = []
            if episode["entered_final_third"]:
                outcome_parts.append("进入进攻三区")
            if episode["shot_within_8s"]:
                outcome_parts.append("8秒内形成射门")
            if episode["retained_possession_after_episode"]:
                outcome_parts.append("继续保持球权")
            outcome = "、".join(outcome_parts) if outcome_parts else "后续收益未确认"
            confidence = min(
                0.9,
                0.5 + float(episode["duration_sec"]) / 12.0 + float(episode["mean_advantage"]) * 0.06 + int(episode["effectiveness_score"]) * 0.04,
            )
            event = TimelineEvent(
                event_id=f"numerical-superiority-{number:03d}",
                event_type="numerical_superiority_candidate",
                frame=int(episode["start_frame"]),
                time_sec=float(episode["start_sec"]),
                team=str(episode["attacking_team"]),
                actor_track_id=None,
                target_track_id=None,
                confidence=round(confidence, 4),
                label_zh="动态多打少候选",
                metrics=episode,
            )
            timeline_events.append(event)
            findings.append(
                Finding(
                    finding_id=f"numerical-superiority-{number:03d}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="numerical_superiority_candidate",
                    title_zh=f"{episode['start_sec']:.2f}秒 {team_label}{ZONE_LABELS[str(episode['zone'])]}多打少候选",
                    summary_zh=(
                        f"持续{float(episode['duration_sec']):.1f}秒，平均{float(episode['mean_attackers']):.1f}对"
                        f"{float(episode['mean_defenders']):.1f}，峰值人数优势{episode['peak_advantage']}人；{outcome}。"
                    ),
                    confidence=event.confidence,
                    start_sec=max(0.0, float(episode["start_sec"]) - 3.0),
                    end_sec=float(episode["end_sec"]) + outcome_window,
                    metrics=episode,
                    evidence=[Evidence(event.frame, event.time_sec, "连续球周局部人数关系及后续结果窗口。", episode)],
                    limitations_zh=["多打少只统计当前可见球员和球周邻域，不代表完整比赛人数结构。"],
                )
            )

        team_summary = {}
        for team in ("left", "right"):
            items = [item for item in episodes if item["attacking_team"] == team]
            team_summary[team] = {
                "episodes": len(items),
                "successful_outcome_proxies": sum(int(item["effectiveness_score"]) > 0 for item in items),
                "shot_outcomes": sum(bool(item["shot_within_8s"]) for item in items),
                "final_third_outcomes": sum(bool(item["entered_final_third"]) for item in items),
                "zone_counts": dict(Counter(str(item["zone"]) for item in items)),
                "mean_peak_advantage": round(sum(int(item["peak_advantage"]) for item in items) / len(items), 3) if items else None,
            }
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "dynamic_ball_neighborhood_superiority_v1", "teams": team_summary, "episodes": episodes},
            findings=findings,
            events=timeline_events,
            warnings=["人数优势需要结合朝向、速度、球场外可见性和后续结果，不自动等同于成功战术。"],
        )
