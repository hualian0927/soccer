"""Relational player graph features inspired by geometric tactical models."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..base import TacticalAnalyzer, register_analyzer
from ..models import HALF_LENGTH, HALF_WIDTH, AnalysisContext, AnalyzerOutput, Evidence, Finding
from .events import nearest_ball_player


def pairwise_mean(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    distances = []
    for index in range(len(points)):
        for other in range(index + 1, len(points)):
            distances.append(float(np.linalg.norm(points[index] - points[other])))
    return float(np.mean(distances)) if distances else 0.0


@register_analyzer("relational_space")
class RelationalSpaceAnalyzer(TacticalAnalyzer):
    priority = "P1"
    description_zh = "把球员视为关系图节点，分析局部人数、队间距离和定位球结构"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        sample_step = max(1, int(self.config.get("sample_step_frames", max(1, round(context.fps / 2.0)))))
        local_radius = float(self.config.get("local_radius_m", 14.0))
        overload_margin = int(self.config.get("overload_margin", 2))
        cooldown_sec = float(self.config.get("candidate_cooldown_sec", 6.0))
        min_corner_players = int(self.config.get("min_corner_players", 6))
        graph_samples = []
        overloads = []
        corner_structures = []
        last_overload: dict[str, float] = {"left": -999.0, "right": -999.0}
        last_corner = -999.0

        for frame in context.frames:
            if (frame.frame - context.frames[0].frame) % sample_step:
                continue
            ball = frame.ball
            if ball is None or not ball.has_pitch_position:
                continue
            possessor, control_distance = nearest_ball_player(frame, float(self.config.get("control_radius_m", 8.0)))
            possession_team = possessor.team if possessor else None
            team_points: dict[str, np.ndarray] = {}
            for team in ("left", "right"):
                unique = {
                    player.track_id: (float(player.pitch_x), float(player.pitch_y))
                    for player in frame.players
                    if player.team == team and player.has_pitch_position
                }
                team_points[team] = np.asarray(list(unique.values()), dtype=np.float32)
            if not len(team_points["left"]) or not len(team_points["right"]):
                continue

            ball_xy = np.asarray([float(ball.pitch_x), float(ball.pitch_y)], dtype=np.float32)
            local_counts = {
                team: int(np.sum(np.linalg.norm(points - ball_xy, axis=1) <= local_radius))
                for team, points in team_points.items()
            }
            nearest_opponent = None
            if possession_team:
                opponent = "right" if possession_team == "left" else "left"
                opponent_distances = np.linalg.norm(team_points[opponent] - ball_xy, axis=1)
                nearest_opponent = float(np.min(opponent_distances)) if len(opponent_distances) else None

            graph_sample = {
                "frame": frame.frame,
                "time_sec": round(frame.time_sec, 3),
                "nodes": int(len(team_points["left"]) + len(team_points["right"])),
                "left_mean_pair_distance_m": round(pairwise_mean(team_points["left"]), 3),
                "right_mean_pair_distance_m": round(pairwise_mean(team_points["right"]), 3),
                "possession_team": possession_team,
                "control_distance_m": round(control_distance, 3) if control_distance is not None else None,
                "local_counts": local_counts,
                "nearest_opponent_m": round(nearest_opponent, 3) if nearest_opponent is not None else None,
            }
            graph_samples.append(graph_sample)

            if possession_team:
                opponent = "right" if possession_team == "left" else "left"
                margin = local_counts[possession_team] - local_counts[opponent]
                if (
                    local_counts[possession_team] >= 3
                    and margin >= overload_margin
                    and frame.time_sec - last_overload[possession_team] >= cooldown_sec
                ):
                    overload = {**graph_sample, "margin": margin, "opponent": opponent}
                    overloads.append(overload)
                    last_overload[possession_team] = frame.time_sec

            near_corner = (
                abs(float(ball.pitch_x)) >= HALF_LENGTH - 7.0
                and abs(float(ball.pitch_y)) >= HALF_WIDTH - 7.0
            )
            if near_corner and frame.time_sec - last_corner >= cooldown_sec:
                target_goal_x = HALF_LENGTH if float(ball.pitch_x) > 0 else -HALF_LENGTH
                in_box = {
                    team: int(
                        np.sum(
                            (np.abs(points[:, 0] - target_goal_x) <= 18.5)
                            & (np.abs(points[:, 1]) <= 22.0)
                        )
                    )
                    for team, points in team_points.items()
                }
                if min(in_box.values()) >= 1 and sum(in_box.values()) >= min_corner_players:
                    corner_structures.append({**graph_sample, "players_in_penalty_area": in_box})
                    last_corner = frame.time_sec

        findings: list[Finding] = []
        for index, overload in enumerate(overloads[:12], 1):
            team = overload["possession_team"]
            label = "左队" if team == "left" else "右队"
            opponent = overload["opponent"]
            local = overload["local_counts"]
            findings.append(
                Finding(
                    finding_id=f"overload-{index:03d}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="local_overload_candidate",
                    title_zh=f"{overload['time_sec']:.2f}秒 {label}局部人数优势候选",
                    summary_zh=(
                        f"球周围{local_radius:.0f}米内，持球队{local[team]}人、对手{local[opponent]}人，"
                        f"人数差为{overload['margin']}。"
                    ),
                    confidence=min(0.90, 0.55 + overload["margin"] * 0.08),
                    start_sec=max(0.0, overload["time_sec"] - 3.0),
                    end_sec=overload["time_sec"] + 4.0,
                    metrics=overload,
                    evidence=[
                        Evidence(
                            frame=overload["frame"],
                            time_sec=overload["time_sec"],
                            description_zh="以球为中心构图后的局部节点计数。",
                            metrics=overload,
                        )
                    ],
                    limitations_zh=["人数优势不自动代表战术成功，需要结合球员朝向、速度和后续结果。"],
                )
            )

        for index, corner in enumerate(corner_structures[:8], 1):
            findings.append(
                Finding(
                    finding_id=f"corner-structure-{index:03d}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="corner_structure_candidate",
                    title_zh=f"{corner['time_sec']:.2f}秒 角球站位结构候选",
                    summary_zh=(
                        "禁区节点分布：左队"
                        f"{corner['players_in_penalty_area']['left']}人，右队"
                        f"{corner['players_in_penalty_area']['right']}人。"
                    ),
                    confidence=0.68,
                    start_sec=max(0.0, corner["time_sec"] - 4.0),
                    end_sec=corner["time_sec"] + 6.0,
                    metrics=corner,
                    evidence=[
                        Evidence(
                            frame=corner["frame"],
                            time_sec=corner["time_sec"],
                            description_zh="角旗区有球且禁区内存在双方球员节点。",
                            metrics=corner,
                        )
                    ],
                    limitations_zh=["当前只描述站位关系，不预测接球人或射门概率。"],
                )
            )

        possession_samples = Counter(sample["possession_team"] for sample in graph_samples)
        summary = {
            "representation": "dynamic_player_relation_graph_v1",
            "sample_step_frames": sample_step,
            "graph_sample_count": len(graph_samples),
            "possession_sample_counts": dict(possession_samples),
            "local_overload_candidates": len(overloads),
            "corner_structure_candidates": len(corner_structures),
            "graph_samples_preview": graph_samples[:20],
        }
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary=summary,
            findings=findings,
            warnings=[
                "关系图目前使用几何规则；后续可在相同接口下替换为 GNN、接球人预测或生成式站位模型。"
            ],
        )
