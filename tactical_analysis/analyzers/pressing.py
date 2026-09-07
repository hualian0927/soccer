"""Pressure episodes and pressing-efficiency proxies."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ..base import TacticalAnalyzer, register_analyzer
from ..models import HALF_LENGTH, AnalysisContext, AnalyzerOutput, Evidence, Finding, TimelineEvent
from .events import ConstrainedPossessionDecoder, attack_direction, ball_point
from .progression import extract_pass_records


def _press_zone(ball_x: float, possessing_direction: int) -> str:
    progress = ball_x * possessing_direction
    if progress < -HALF_LENGTH / 3.0:
        return "high_press"
    if progress > HALF_LENGTH / 3.0:
        return "low_press"
    return "mid_press"


@register_analyzer("pressing_analysis")
class PressingAnalyzer(TacticalAnalyzer):
    priority = "P3"
    description_zh = "识别压迫回合、参与人数、施压距离与迫使回传/夺回球权结果"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        decoder = ConstrainedPossessionDecoder(
            float(self.config.get("control_radius_m", 7.5)),
            int(self.config.get("possession_confirm_samples", 3)),
            int(self.config.get("possession_release_samples", 4)),
        )
        pressure_radius = float(self.config.get("pressure_radius_m", 8.0))
        active_distance = float(self.config.get("active_distance_m", 6.0))
        sample_step = max(1, int(self.config.get("sample_step_frames", 5)))
        path = [(frame, decoder.update(frame)) for frame in context.frames]
        samples: list[dict[str, Any]] = []
        for index, (frame, possession) in enumerate(path):
            if index % sample_step or possession.team not in {"left", "right"}:
                continue
            point = ball_point(frame)
            if point is None:
                continue
            defending_team = "right" if possession.team == "left" else "left"
            distances = []
            for player in frame.players:
                if player.team != defending_team or not player.has_pitch_position:
                    continue
                distances.append(math.hypot(float(player.pitch_x) - point[0], float(player.pitch_y) - point[1]))
            if not distances:
                continue
            participants = sum(distance <= pressure_radius for distance in distances)
            nearest = min(distances)
            if nearest > active_distance or participants == 0:
                continue
            samples.append(
                {
                    "frame": frame.frame,
                    "time_sec": frame.time_sec,
                    "pressing_team": defending_team,
                    "possessing_team": possession.team,
                    "participants": participants,
                    "nearest_distance_m": nearest,
                    "zone": _press_zone(point[0], attack_direction(frame, possession.team)),
                    "ball_progress": point[0] * attack_direction(frame, possession.team),
                    "ball_y": point[1],
                }
            )

        episodes: list[dict[str, Any]] = []
        active: list[dict[str, Any]] = []
        maximum_gap = float(self.config.get("maximum_episode_gap_sec", 0.65))
        for sample in samples:
            if active and (
                sample["pressing_team"] != active[-1]["pressing_team"]
                or sample["time_sec"] - active[-1]["time_sec"] > maximum_gap
            ):
                episodes.append(self._summarize_episode(active, path))
                active = []
            active.append(sample)
        if active:
            episodes.append(self._summarize_episode(active, path))

        minimum_duration = float(self.config.get("minimum_episode_sec", 0.4))
        episodes = [episode for episode in episodes if episode["duration_sec"] >= minimum_duration]
        findings: list[Finding] = []
        timeline_events: list[TimelineEvent] = []
        ranked = sorted(
            episodes,
            key=lambda item: (item["success_score"], item["mean_participants"], item["duration_sec"]),
            reverse=True,
        )
        for number, episode in enumerate(ranked[: int(self.config.get("max_findings", 8))], 1):
            label = "左队" if episode["pressing_team"] == "left" else "右队"
            zone_zh = {"high_press": "高位", "mid_press": "中位", "low_press": "低位"}[episode["zone"]]
            outcome = "夺回球权" if episode["regained_within_5s"] else ("迫使向后" if episode["forced_backward"] else "未确认收益")
            confidence = min(0.9, 0.48 + episode["duration_sec"] / 12.0 + episode["mean_participants"] * 0.05 + episode["success_score"] * 0.12)
            finding = Finding(
                finding_id=f"pressing-{number:03d}",
                analyzer=self.name,
                priority=self.priority,
                category="pressing_episode_candidate",
                title_zh=f"{episode['start_sec']:.2f}秒 {label}{zone_zh}压迫候选",
                summary_zh=(
                    f"持续{episode['duration_sec']:.1f}秒，平均{episode['mean_participants']:.1f}人参与，"
                    f"最近施压距离{episode['minimum_distance_m']:.1f}米，"
                    f"方向为{episode['pressing_direction_zh']}，结果代理：{outcome}。"
                ),
                confidence=round(confidence, 4),
                start_sec=max(0.0, episode["start_sec"] - 2.0),
                end_sec=episode["end_sec"] + 5.0,
                metrics=episode,
                evidence=[Evidence(episode["start_frame"], episode["start_sec"], "连续球周防守距离样本。", episode)],
                limitations_zh=["压迫收益只使用回传与球权切换代理，尚未识别封堵传球角度和犯规。"],
            )
            findings.append(finding)
            timeline_events.append(
                TimelineEvent(
                    event_id=f"pressing-{number:03d}",
                    event_type="high_press_candidate" if episode["zone"] == "high_press" else "pressing_episode_candidate",
                    frame=int(episode["start_frame"]),
                    time_sec=float(episode["start_sec"]),
                    team=str(episode["pressing_team"]),
                    actor_track_id=None,
                    target_track_id=None,
                    confidence=round(confidence, 4),
                    label_zh=f"{zone_zh}压迫候选",
                    metrics=episode,
                )
            )

        by_team = {}
        pass_records = extract_pass_records(context, self.config)
        frame_by_number = {frame.frame: frame for frame in context.frames}
        for team in ("left", "right"):
            team_episodes = [episode for episode in episodes if episode["pressing_team"] == team]
            opponent = "right" if team == "left" else "left"
            opponent_build_up_passes = 0
            for record in pass_records:
                if record["team"] != opponent:
                    continue
                frame = frame_by_number.get(int(record["start_frame"]))
                direction = attack_direction(frame, opponent) if frame is not None else (1 if opponent == "left" else -1)
                if float(record["start_x"]) * direction <= 10.5:
                    opponent_build_up_passes += 1
            defensive_actions = sum(episode["zone"] == "high_press" for episode in team_episodes)
            by_team[team] = {
                "episodes": len(team_episodes),
                "successful_proxies": sum(episode["success_score"] > 0 for episode in team_episodes),
                "efficiency_proxy": round(
                    sum(episode["success_score"] > 0 for episode in team_episodes) / max(len(team_episodes), 1),
                    4,
                ),
                "zone_counts": dict(Counter(episode["zone"] for episode in team_episodes)),
                "opponent_build_up_passes": opponent_build_up_passes,
                "high_press_defensive_action_proxies": defensive_actions,
                "ppda_proxy": round(opponent_build_up_passes / defensive_actions, 3) if defensive_actions else None,
                "mean_intensity_proxy": round(
                    sum(episode["intensity_proxy"] for episode in team_episodes) / max(len(team_episodes), 1),
                    3,
                ),
            }
            high_episodes = [episode for episode in team_episodes if episode["zone"] == "high_press"]
            if high_episodes:
                representative = max(
                    high_episodes,
                    key=lambda item: (item["success_score"], item["mean_participants"], item["duration_sec"]),
                )
                ppda = by_team[team]["ppda_proxy"]
                findings.append(
                    Finding(
                        finding_id=f"high-press-summary-{team}",
                        analyzer=self.name,
                        priority=self.priority,
                        category="high_press_ppda_summary",
                        title_zh=f"{'左队' if team == 'left' else '右队'}高位逼抢与PPDA代理",
                        summary_zh=(
                            f"识别{len(high_episodes)}个高位压迫回合，PPDA视觉代理为"
                            f"{f'{ppda:.2f}' if ppda is not None else '样本不足'}，"
                            f"压迫收益代理率{by_team[team]['efficiency_proxy']:.0%}。"
                        ),
                        confidence=round(min(0.88, 0.5 + len(high_episodes) * 0.04), 4),
                        start_sec=float(representative["start_sec"]),
                        end_sec=float(representative["end_sec"]) + 5.0,
                        metrics={**by_team[team], "representative_episode": representative},
                        evidence=[Evidence(int(representative["start_frame"]), float(representative["start_sec"]), "高位压迫代表回合及球队聚合。", representative)],
                        limitations_zh=["PPDA未覆盖全部传球与防守动作，只能作为同场粗粒度强度代理。"],
                    )
                )
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "ball_pressure_episode_proxy_v1", "teams": by_team, "episodes": episodes},
            findings=findings,
            events=timeline_events,
            warnings=["PPDA为视觉代理且未识别所有防守动作；压迫质量优先阅读参与、距离、方向和结果。"],
        )

    @staticmethod
    def _summarize_episode(samples: list[dict[str, Any]], path: list[tuple]) -> dict[str, Any]:
        first, last = samples[0], samples[-1]
        regained = False
        for _, possession in path:
            if possession.time_sec < first["time_sec"] + 0.5:
                continue
            if possession.time_sec > last["time_sec"] + 5.0:
                break
            if possession.team == first["pressing_team"]:
                regained = True
                break
        forced_backward = last["ball_progress"] <= first["ball_progress"] - 3.0
        mean_ball_y = sum(float(sample["ball_y"]) for sample in samples) / len(samples)
        if mean_ball_y >= 10.0:
            pressing_direction = "upper_touchline"
            pressing_direction_zh = "压向上侧边线"
        elif mean_ball_y <= -10.0:
            pressing_direction = "lower_touchline"
            pressing_direction_zh = "压向下侧边线"
        else:
            pressing_direction = "central"
            pressing_direction_zh = "中路合围"
        trigger = "opponent_build_up" if first["zone"] == "high_press" else (
            "middle_third_control" if first["zone"] == "mid_press" else "deep_defensive_pressure"
        )
        duration = max(0.0, last["time_sec"] - first["time_sec"])
        mean_participants = sum(sample["participants"] for sample in samples) / len(samples)
        minimum_distance = min(sample["nearest_distance_m"] for sample in samples)
        return {
            "start_frame": first["frame"],
            "start_sec": round(first["time_sec"], 3),
            "end_sec": round(last["time_sec"], 3),
            "duration_sec": round(duration, 3),
            "pressing_team": first["pressing_team"],
            "possessing_team": first["possessing_team"],
            "zone": Counter(sample["zone"] for sample in samples).most_common(1)[0][0],
            "mean_participants": round(mean_participants, 3),
            "peak_participants": max(sample["participants"] for sample in samples),
            "minimum_distance_m": round(minimum_distance, 3),
            "pressing_direction": pressing_direction,
            "pressing_direction_zh": pressing_direction_zh,
            "pressing_trigger_proxy": trigger,
            "intensity_proxy": round(mean_participants / max(minimum_distance, 1.0) * max(duration, 0.2), 3),
            "forced_backward": forced_backward,
            "regained_within_5s": regained,
            "success_score": int(forced_backward) + int(regained),
        }
