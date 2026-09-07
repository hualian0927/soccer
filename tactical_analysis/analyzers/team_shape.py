"""Phase-conditioned team width, length, line height and compactness engine."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..base import TacticalAnalyzer, register_analyzer
from ..models import HALF_LENGTH, AnalysisContext, AnalyzerOutput, Evidence, Finding, FrameState
from .events import ConstrainedPossessionDecoder, attack_direction


def _convex_hull_area(points: np.ndarray) -> float:
    """Return a dependency-free 2D monotonic-chain hull area."""

    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) < 3:
        return 0.0

    def cross(origin, left, right) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return abs(
        sum(
            hull[index][0] * hull[(index + 1) % len(hull)][1]
            - hull[(index + 1) % len(hull)][0] * hull[index][1]
            for index in range(len(hull))
        )
    ) / 2.0


def shape_sample(frame: FrameState, team: str, state: str, minimum_players: int) -> dict[str, Any] | None:
    unique = {
        player.track_id: player
        for player in frame.players
        if player.team == team and player.has_pitch_position
    }
    players = list(unique.values())
    if len(players) < minimum_players:
        return None
    points = np.asarray(
        [(float(player.pitch_x), float(player.pitch_y)) for player in players],
        dtype=np.float32,
    )
    direction = attack_direction(frame, team)
    outfield = [player for player in players if player.role != "goalkeeper"]
    if len(outfield) < 6:
        return None
    progress = np.sort(np.asarray([float(player.pitch_x) * direction for player in outfield]))
    groups = [chunk for chunk in np.array_split(progress, 3) if len(chunk)]
    if len(groups) != 3:
        return None
    centers = [float(np.median(group)) for group in groups]
    return {
        "frame": frame.frame,
        "time_sec": round(frame.time_sec, 3),
        "state": state,
        "visible_players": len(players),
        "width_m": round(float(np.ptp(points[:, 1])), 3),
        "length_m": round(float(np.ptp(points[:, 0])), 3),
        "centroid_height_m": round(float(np.mean(points[:, 0] * direction) + HALF_LENGTH), 3),
        "defensive_line_height_m": round(centers[0] + HALF_LENGTH, 3),
        "midfield_line_height_m": round(centers[1] + HALF_LENGTH, 3),
        "forward_line_height_m": round(centers[2] + HALF_LENGTH, 3),
        "defense_midfield_gap_m": round(centers[1] - centers[0], 3),
        "midfield_forward_gap_m": round(centers[2] - centers[1], 3),
        "convex_hull_area_m2": round(_convex_hull_area(points), 3),
    }


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "width_m",
        "length_m",
        "centroid_height_m",
        "defensive_line_height_m",
        "midfield_line_height_m",
        "forward_line_height_m",
        "defense_midfield_gap_m",
        "midfield_forward_gap_m",
        "convex_hull_area_m2",
    ]
    if not samples:
        return {"sample_count": 0}
    result: dict[str, Any] = {"sample_count": len(samples)}
    for name in metrics:
        values = np.asarray([float(sample[name]) for sample in samples], dtype=np.float32)
        result[f"median_{name}"] = round(float(np.median(values)), 3)
        result[f"p25_{name}"] = round(float(np.percentile(values, 25)), 3)
        result[f"p75_{name}"] = round(float(np.percentile(values, 75)), 3)
    median_width = result["median_width_m"]
    median_length = result["median_length_m"]
    representative = min(
        samples,
        key=lambda sample: abs(sample["width_m"] - median_width) + abs(sample["length_m"] - median_length),
    )
    result["representative_frame"] = representative
    return result


@register_analyzer("team_shape_engine")
class TeamShapeAnalyzer(TacticalAnalyzer):
    priority = "P1"
    description_zh = "按有球/无球状态分析宽度、纵深、三线高度、线距和紧凑度"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        decoder = ConstrainedPossessionDecoder(
            float(self.config.get("control_radius_m", 7.5)),
            int(self.config.get("possession_confirm_samples", 3)),
            int(self.config.get("possession_release_samples", 4)),
        )
        sample_step = max(1, int(self.config.get("sample_step_frames", 10)))
        minimum_players = int(self.config.get("minimum_visible_players", 7))
        samples: dict[str, dict[str, list[dict[str, Any]]]] = {
            "left": defaultdict(list),
            "right": defaultdict(list),
        }
        for index, frame in enumerate(context.frames):
            possession = decoder.update(frame)
            if index % sample_step:
                continue
            for team in ("left", "right"):
                if possession.team == team:
                    state = "in_possession"
                elif possession.team in {"left", "right"}:
                    state = "out_of_possession"
                else:
                    continue
                sample = shape_sample(frame, team, state, minimum_players)
                if sample:
                    samples[team][state].append(sample)

        team_summary: dict[str, Any] = {}
        findings: list[Finding] = []
        warnings: list[str] = []
        for team in ("left", "right"):
            states = {
                state: _aggregate(samples[team].get(state, []))
                for state in ("in_possession", "out_of_possession")
            }
            team_summary[team] = {"states": states}
            usable = [value for value in states.values() if value.get("sample_count", 0)]
            if not usable:
                warnings.append(f"{team} 队缺少至少{minimum_players}人可见的完整队形帧。")
                continue
            label = "左队" if team == "left" else "右队"
            in_shape = states["in_possession"]
            out_shape = states["out_of_possession"]
            primary = in_shape if in_shape.get("sample_count", 0) else out_shape
            representative = primary["representative_frame"]
            comparison = ""
            if in_shape.get("sample_count", 0) and out_shape.get("sample_count", 0):
                comparison = (
                    f"有球宽度{in_shape['median_width_m']:.1f}米、纵深{in_shape['median_length_m']:.1f}米；"
                    f"无球宽度{out_shape['median_width_m']:.1f}米、纵深{out_shape['median_length_m']:.1f}米。"
                )
            else:
                comparison = (
                    f"当前有效状态中位宽度{primary['median_width_m']:.1f}米、"
                    f"纵深{primary['median_length_m']:.1f}米。"
                )
            confidence = min(0.92, 0.5 + sum(item.get("sample_count", 0) for item in usable) / 160.0)
            findings.append(
                Finding(
                    finding_id=f"team-shape-{team}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="team_shape_by_possession",
                    title_zh=f"{label}有球/无球队形结构",
                    summary_zh=(
                        comparison
                        + f"代表状态后卫—中场线距{primary['median_defense_midfield_gap_m']:.1f}米、"
                        f"中场—前锋线距{primary['median_midfield_forward_gap_m']:.1f}米。"
                    ),
                    confidence=round(confidence, 4),
                    start_sec=representative["time_sec"],
                    end_sec=representative["time_sec"],
                    metrics=team_summary[team],
                    evidence=[
                        Evidence(
                            representative["frame"],
                            representative["time_sec"],
                            "最接近该状态中位宽度和纵深的代表帧。",
                            representative,
                        )
                    ],
                    limitations_zh=[
                        "三线由可见外场球员的纵向聚类近似，转播视野不完整时会低估宽度和纵深。",
                        "紧凑度使用二维凸包面积，不包含球员朝向和不可见球员。",
                    ],
                )
            )
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "phase_conditioned_team_shape_v1", "teams": team_summary},
            findings=findings,
            warnings=warnings,
        )
