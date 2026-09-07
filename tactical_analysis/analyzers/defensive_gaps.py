"""Exploitable defensive-gap candidates between team lines and channels."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from ..base import TacticalAnalyzer, register_analyzer
from ..models import HALF_LENGTH, HALF_WIDTH, AnalysisContext, AnalyzerOutput, Evidence, Finding, FrameState, TimelineEvent
from .events import ConstrainedPossessionDecoder, EventTimelineAnalyzer, PossessionSample, attack_direction, ball_point


def gap_channel(y: float) -> str:
    if y > HALF_WIDTH / 3.0:
        return "upper_flank"
    if y < -HALF_WIDTH / 3.0:
        return "lower_flank"
    if abs(y) >= 7.0:
        return "half_space"
    return "central"


CHANNEL_LABELS = {
    "upper_flank": "上方边路",
    "lower_flank": "下方边路",
    "half_space": "肋部",
    "central": "中路",
}


def defensive_gap_sample(
    frame: FrameState,
    possession: PossessionSample,
    minimum_visible_defenders: int,
    minimum_line_gap: float,
    pressure_radius: float,
) -> dict[str, Any] | None:
    if possession.team not in {"left", "right"}:
        return None
    point = ball_point(frame)
    if point is None:
        return None
    attacking_team = str(possession.team)
    defending_team = "right" if attacking_team == "left" else "left"
    direction = attack_direction(frame, attacking_team)
    defenders = {
        player.track_id: player
        for player in frame.players
        if player.team == defending_team and player.role != "goalkeeper" and player.has_pitch_position
    }
    attackers = {
        player.track_id: player
        for player in frame.players
        if player.team == attacking_team and player.role != "goalkeeper" and player.has_pitch_position
    }
    if len(defenders) < minimum_visible_defenders or not attackers:
        return None
    defender_progress = np.asarray(
        sorted(float(player.pitch_x) * direction for player in defenders.values()),
        dtype=np.float32,
    )
    groups = [chunk for chunk in np.array_split(defender_progress, 3) if len(chunk) >= 2]
    if len(groups) < 3:
        return None
    line_centers = sorted(float(np.median(group)) for group in groups)
    line_gaps = [line_centers[index + 1] - line_centers[index] for index in range(len(line_centers) - 1)]
    largest_index = int(np.argmax(line_gaps))
    largest_gap = float(line_gaps[largest_index])
    gap_start, gap_end = line_centers[largest_index], line_centers[largest_index + 1]
    ball_progress = point[0] * direction
    ball_between_lines = gap_start - 1.0 <= ball_progress <= gap_end + 1.0
    defender_distances = [
        math.hypot(float(player.pitch_x) - point[0], float(player.pitch_y) - point[1])
        for player in defenders.values()
    ]
    attacker_distances = [
        math.hypot(float(player.pitch_x) - point[0], float(player.pitch_y) - point[1])
        for player in attackers.values()
    ]
    defenders_near_ball = sum(distance <= pressure_radius for distance in defender_distances)
    attackers_near_ball = sum(distance <= pressure_radius + 2.0 for distance in attacker_distances)
    nearest_defender = min(defender_distances)
    advanced_enough = ball_progress >= -HALF_LENGTH / 3.0
    spatially_exposed = defenders_near_ball <= 1 and nearest_defender >= 4.0
    if not advanced_enough or largest_gap < minimum_line_gap or not ball_between_lines or not spatially_exposed:
        return None
    return {
        "frame": frame.frame,
        "time_sec": frame.time_sec,
        "attacking_team": attacking_team,
        "defending_team": defending_team,
        "ball_x": point[0],
        "ball_y": point[1],
        "ball_progress": ball_progress,
        "channel": gap_channel(point[1]),
        "line_centers_m": [round(value, 3) for value in line_centers],
        "largest_line_gap_m": round(largest_gap, 3),
        "gap_start_m": round(gap_start, 3),
        "gap_end_m": round(gap_end, 3),
        "defenders_near_ball": defenders_near_ball,
        "attackers_near_ball": attackers_near_ball,
        "nearest_defender_m": round(nearest_defender, 3),
        "visible_defenders": len(defenders),
        "visible_attackers": len(attackers),
    }


@register_analyzer("defensive_gaps")
class DefensiveGapAnalyzer(TacticalAnalyzer):
    priority = "P3"
    description_zh = "识别对手控球时可利用的线间空档和低防守密度通道"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        decoder = ConstrainedPossessionDecoder(
            float(self.config.get("control_radius_m", 7.5)),
            int(self.config.get("possession_confirm_samples", 3)),
            int(self.config.get("possession_release_samples", 4)),
        )
        sample_step = max(1, int(self.config.get("sample_step_frames", 5)))
        minimum_visible = int(self.config.get("minimum_visible_defenders", 7))
        minimum_line_gap = float(self.config.get("minimum_line_gap_m", 12.0))
        pressure_radius = float(self.config.get("pressure_radius_m", 8.0))
        maximum_gap = float(self.config.get("maximum_episode_gap_sec", 0.65))
        minimum_duration = float(self.config.get("minimum_episode_sec", 0.3))
        outcome_window = float(self.config.get("outcome_window_sec", 8.0))
        path = [(frame, decoder.update(frame)) for frame in context.frames]
        samples = []
        for index, (frame, possession) in enumerate(path):
            if index % sample_step:
                continue
            sample = defensive_gap_sample(
                frame,
                possession,
                minimum_visible,
                minimum_line_gap,
                pressure_radius,
            )
            if sample is not None:
                samples.append(sample)

        raw_episodes: list[list[dict[str, Any]]] = []
        active: list[dict[str, Any]] = []
        for sample in samples:
            if active and (
                sample["defending_team"] != active[-1]["defending_team"]
                or sample["channel"] != active[-1]["channel"]
                or sample["time_sec"] - active[-1]["time_sec"] > maximum_gap
            ):
                raw_episodes.append(active)
                active = []
            active.append(sample)
        if active:
            raw_episodes.append(active)

        shots = [
            event
            for event in EventTimelineAnalyzer(self.config).analyze(context).events
            if event.event_type == "shot_candidate"
        ]
        episodes: list[dict[str, Any]] = []
        for group in raw_episodes:
            duration = group[-1]["time_sec"] - group[0]["time_sec"]
            if duration < minimum_duration:
                continue
            attacking_team = group[0]["attacking_team"]
            shot = next(
                (
                    event for event in shots
                    if event.team == attacking_team and group[0]["time_sec"] <= event.time_sec <= group[-1]["time_sec"] + outcome_window
                ),
                None,
            )
            max_progress = max(float(sample["ball_progress"]) for sample in group)
            entered_final_third = max_progress >= HALF_LENGTH / 3.0
            representative = max(
                group,
                key=lambda sample: (
                    float(sample["largest_line_gap_m"]),
                    float(sample["nearest_defender_m"]),
                ),
            )
            severity = (
                float(representative["largest_line_gap_m"]) / max(minimum_line_gap, 1.0)
                + float(representative["nearest_defender_m"]) / max(pressure_radius, 1.0)
                + int(entered_final_third) * 0.5
                + int(shot is not None)
            )
            episodes.append(
                {
                    "start_frame": group[0]["frame"],
                    "representative_frame": representative["frame"],
                    "start_sec": round(float(group[0]["time_sec"]), 3),
                    "end_sec": round(float(group[-1]["time_sec"]), 3),
                    "duration_sec": round(duration, 3),
                    "attacking_team": attacking_team,
                    "defending_team": group[0]["defending_team"],
                    "channel": group[0]["channel"],
                    "maximum_line_gap_m": representative["largest_line_gap_m"],
                    "maximum_nearest_defender_m": representative["nearest_defender_m"],
                    "minimum_defenders_near_ball": min(int(sample["defenders_near_ball"]) for sample in group),
                    "maximum_attackers_near_ball": max(int(sample["attackers_near_ball"]) for sample in group),
                    "entered_final_third": entered_final_third,
                    "shot_within_8s": shot is not None,
                    "shot_time_sec": shot.time_sec if shot else None,
                    "severity_proxy": round(severity, 3),
                    "representative_sample": representative,
                    "sample_count": len(group),
                }
            )

        ranked = sorted(
            episodes,
            key=lambda item: (item["shot_within_8s"], item["severity_proxy"], item["duration_sec"]),
            reverse=True,
        )
        findings: list[Finding] = []
        timeline_events: list[TimelineEvent] = []
        for number, episode in enumerate(ranked[: int(self.config.get("max_findings", 8))], 1):
            defending_label = "左队" if episode["defending_team"] == "left" else "右队"
            outcome = "8秒内被对手形成射门" if episode["shot_within_8s"] else (
                "对手进入进攻三区" if episode["entered_final_third"] else "后续结果未确认"
            )
            confidence = min(
                0.9,
                0.45 + float(episode["duration_sec"]) / 10.0 + float(episode["severity_proxy"]) * 0.08,
            )
            event = TimelineEvent(
                event_id=f"defensive-gap-{number:03d}",
                event_type="defensive_gap_candidate",
                frame=int(episode["representative_frame"]),
                time_sec=float(episode["start_sec"]),
                team=str(episode["defending_team"]),
                actor_track_id=None,
                target_track_id=None,
                confidence=round(confidence, 4),
                label_zh="防守空档候选",
                metrics=episode,
            )
            timeline_events.append(event)
            findings.append(
                Finding(
                    finding_id=f"defensive-gap-{number:03d}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="defensive_gap_candidate",
                    title_zh=f"{episode['start_sec']:.2f}秒 {defending_label}{CHANNEL_LABELS[str(episode['channel'])]}防守空档候选",
                    summary_zh=(
                        f"线间最大空档约{float(episode['maximum_line_gap_m']):.1f}米，球周最近防守人约"
                        f"{float(episode['maximum_nearest_defender_m']):.1f}米；{outcome}。"
                    ),
                    confidence=event.confidence,
                    start_sec=max(0.0, float(episode["start_sec"]) - 3.0),
                    end_sec=float(episode["end_sec"]) + outcome_window,
                    metrics=episode,
                    evidence=[Evidence(event.frame, event.time_sec, "对手控球、三线间距和球周防守密度联合证据。", episode)],
                    limitations_zh=["防守空档只反映可见二维站位，不进行个人失位责任归因。"],
                )
            )

        team_summary = {}
        for team in ("left", "right"):
            items = [item for item in episodes if item["defending_team"] == team]
            team_summary[team] = {
                "gap_episodes": len(items),
                "shot_conceding_outcomes": sum(bool(item["shot_within_8s"]) for item in items),
                "final_third_outcomes": sum(bool(item["entered_final_third"]) for item in items),
                "channel_counts": dict(Counter(str(item["channel"]) for item in items)),
                "mean_maximum_line_gap_m": round(sum(float(item["maximum_line_gap_m"]) for item in items) / len(items), 3) if items else None,
            }
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "line_gap_and_local_pressure_proxy_v1", "teams": team_summary, "episodes": episodes},
            findings=findings,
            events=timeline_events,
            warnings=["空档候选用于定位教练复核片段，不等同于防守失误或失球责任。"],
        )
