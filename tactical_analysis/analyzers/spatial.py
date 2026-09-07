"""P0 spatial occupancy and team-shape summaries."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..base import TacticalAnalyzer, register_analyzer
from ..models import HALF_LENGTH, HALF_WIDTH, AnalysisContext, AnalyzerOutput, Evidence, Finding


def pitch_zone(x: float, y: float) -> tuple[str, str]:
    if x < -HALF_LENGTH / 3.0:
        third = "left_third"
    elif x > HALF_LENGTH / 3.0:
        third = "right_third"
    else:
        third = "middle_third"
    if y < -HALF_WIDTH / 3.0:
        channel = "lower_wing"
    elif y > HALF_WIDTH / 3.0:
        channel = "upper_wing"
    else:
        channel = "central"
    return third, channel


def zh_zone(zone: str) -> str:
    return {
        "left_third": "左侧三区",
        "middle_third": "中场三区",
        "right_third": "右侧三区",
        "lower_wing": "下方边路",
        "central": "中路通道",
        "upper_wing": "上方边路",
    }.get(zone, zone)


@register_analyzer("spatial_structure")
class SpatialStructureAnalyzer(TacticalAnalyzer):
    priority = "P0"
    description_zh = "统计区域占用、平均站位、宽度、纵深和队形面积"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        zone_counts: dict[str, Counter] = {"left": Counter(), "right": Counter()}
        track_points: dict[tuple[str, int], list[tuple[float, float, int]]] = defaultdict(list)
        frame_shapes: dict[str, list[dict[str, float]]] = {"left": [], "right": []}

        for frame in context.frames:
            for team in ("left", "right"):
                players = [
                    player
                    for player in frame.players
                    if player.team == team and player.has_pitch_position
                ]
                unique = {}
                for player in players:
                    unique[player.track_id] = player
                points = np.array(
                    [(float(player.pitch_x), float(player.pitch_y)) for player in unique.values()],
                    dtype=np.float32,
                )
                if len(points) >= 3:
                    width = float(np.ptp(points[:, 1]))
                    depth = float(np.ptp(points[:, 0]))
                    frame_shapes[team].append(
                        {
                            "frame": float(frame.frame),
                            "width_m": width,
                            "depth_m": depth,
                            "area_proxy_m2": width * depth,
                            "centroid_x": float(points[:, 0].mean()),
                            "centroid_y": float(points[:, 1].mean()),
                            "visible_players": float(len(points)),
                        }
                    )
                for player in unique.values():
                    x, y = float(player.pitch_x), float(player.pitch_y)
                    third, channel = pitch_zone(x, y)
                    zone_counts[team][third] += 1
                    zone_counts[team][channel] += 1
                    track_points[(team, player.track_id)].append((x, y, frame.frame))

        team_summary: dict[str, dict] = {}
        findings: list[Finding] = []
        for team in ("left", "right"):
            third_total = sum(zone_counts[team][name] for name in ("left_third", "middle_third", "right_third"))
            channel_total = sum(zone_counts[team][name] for name in ("lower_wing", "central", "upper_wing"))
            thirds = {
                name: round(zone_counts[team][name] / max(third_total, 1), 4)
                for name in ("left_third", "middle_third", "right_third")
            }
            channels = {
                name: round(zone_counts[team][name] / max(channel_total, 1), 4)
                for name in ("lower_wing", "central", "upper_wing")
            }
            shapes = frame_shapes[team]
            median_width = float(np.median([shape["width_m"] for shape in shapes])) if shapes else 0.0
            median_depth = float(np.median([shape["depth_m"] for shape in shapes])) if shapes else 0.0
            median_area = float(np.median([shape["area_proxy_m2"] for shape in shapes])) if shapes else 0.0
            top_third = max(thirds, key=thirds.get)
            top_channel = max(channels, key=channels.get)
            averages = []
            for (track_team, track_id), points in track_points.items():
                if track_team != team or len(points) < int(self.config.get("min_track_samples", 8)):
                    continue
                averages.append(
                    {
                        "track_id": track_id,
                        "samples": len(points),
                        "mean_x": round(float(np.mean([point[0] for point in points])), 3),
                        "mean_y": round(float(np.mean([point[1] for point in points])), 3),
                    }
                )
            team_summary[team] = {
                "third_occupancy": thirds,
                "channel_occupancy": channels,
                "median_width_m": round(median_width, 3),
                "median_depth_m": round(median_depth, 3),
                "median_area_proxy_m2": round(median_area, 3),
                "shape_frame_count": len(shapes),
                "average_positions": sorted(averages, key=lambda item: item["samples"], reverse=True),
            }
            support = min(1.0, len(shapes) / max(context.fps * 4.0, 1.0))
            confidence = 0.45 + 0.45 * support
            label = "左队" if team == "left" else "右队"
            shape_label = "紧凑" if median_area < 900 else ("均衡" if median_area < 1550 else "拉开")
            findings.append(
                Finding(
                    finding_id=f"spatial-{team}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="team_shape",
                    title_zh=f"{label}主要占用{zh_zone(top_third)}与{zh_zone(top_channel)}",
                    summary_zh=(
                        f"中位宽度{median_width:.1f}米、纵深{median_depth:.1f}米，"
                        f"整体队形倾向为{shape_label}。"
                    ),
                    confidence=round(confidence, 4),
                    start_sec=context.frames[0].time_sec if context.frames else None,
                    end_sec=context.duration_sec,
                    metrics=team_summary[team],
                    evidence=[
                        Evidence(
                            frame=None,
                            time_sec=None,
                            description_zh=f"基于{len(shapes)}个有效阵地帧的时间聚合，而非单帧判断。",
                            metrics={
                                "top_third_ratio": thirds[top_third],
                                "top_channel_ratio": channels[top_channel],
                            },
                        )
                    ],
                    limitations_zh=["区域占用受转播镜头覆盖范围影响，不等同于完整比赛跑位热图。"],
                )
            )

        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"teams": team_summary},
            findings=findings,
            warnings=["左右三区是统一场地坐标，不自动等同于某队的进攻或防守三区。"],
        )
