#!/usr/bin/env python3
"""Generate first-pass football tactical analytics from SoccerNetGSR JSON.

This script is intentionally rule-based and lightweight. It reuses the local
project's existing outputs: player/ball/referee tracks, team labels, and pitch
coordinates. It does not modify the detection pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2
HALF_WIDTH = PITCH_WIDTH / 2
DEFAULT_FPS = 25.0


@dataclass
class Detection:
    frame: int
    track_id: int
    role: str
    team: str | None
    x: float
    y: float


def parse_frame(image_id: str) -> int:
    return int(str(image_id)[-6:])


def normalize_team(value) -> str | None:
    if value is None:
        return None
    text = str(value).lower()
    if text in {"left", "right"}:
        return text
    return None


def load_detections(json_path: Path) -> list[Detection]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    detections = []
    for pred in data.get("predictions", []):
        pitch = pred.get("bbox_pitch") or {}
        if "x_bottom_middle" not in pitch or "y_bottom_middle" not in pitch:
            continue
        x = float(pitch["x_bottom_middle"])
        y = float(pitch["y_bottom_middle"])
        if not (-HALF_LENGTH - 8 <= x <= HALF_LENGTH + 8 and -HALF_WIDTH - 5 <= y <= HALF_WIDTH + 5):
            continue
        attrs = pred.get("attributes", {})
        detections.append(
            Detection(
                frame=parse_frame(pred.get("image_id", "0")),
                track_id=int(pred["track_id"]),
                role=str(attrs.get("role", "")).lower(),
                team=normalize_team(attrs.get("team")),
                x=x,
                y=y,
            )
        )
    return detections


def draw_pitch(ax, title: str | None = None) -> None:
    ax.set_facecolor("#2f7d32")
    line = "#f8fafc"
    ax.plot([-HALF_LENGTH, HALF_LENGTH, HALF_LENGTH, -HALF_LENGTH, -HALF_LENGTH],
            [-HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH],
            color=line, linewidth=2)
    ax.axvline(0, color=line, linewidth=1.5)
    center = plt.Circle((0, 0), 9.15, color=line, fill=False, linewidth=1.5)
    ax.add_patch(center)
    ax.scatter([0], [0], s=10, color=line)
    # Penalty areas and six-yard boxes.
    for sign in [-1, 1]:
        goal_x = sign * HALF_LENGTH
        pa_x = goal_x - sign * 16.5
        six_x = goal_x - sign * 5.5
        ax.plot([goal_x, pa_x, pa_x, goal_x],
                [-20.16, -20.16, 20.16, 20.16],
                color=line, linewidth=1.2)
        ax.plot([goal_x, six_x, six_x, goal_x],
                [-9.16, -9.16, 9.16, 9.16],
                color=line, linewidth=1.2)
    ax.set_xlim(-HALF_LENGTH - 3, HALF_LENGTH + 3)
    ax.set_ylim(-HALF_WIDTH - 3, HALF_WIDTH + 3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Pitch X (m)")
    ax.set_ylabel("Pitch Y (m)")
    if title:
        ax.set_title(title)


def save_heatmap(detections: list[Detection], output: Path, title: str) -> None:
    xs = [det.x for det in detections]
    ys = [det.y for det in detections]
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax, title)
    if xs:
        heat = ax.hist2d(xs, ys, bins=[42, 28], range=[[-HALF_LENGTH, HALF_LENGTH], [-HALF_WIDTH, HALF_WIDTH]],
                         cmap="inferno", alpha=0.78)
        fig.colorbar(heat[3], ax=ax, shrink=0.75, label="detections")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_average_positions(players: list[Detection], output: Path) -> list[dict]:
    colors = {"left": "#ef4444", "right": "#2563eb"}
    rows = []
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax, "Average Positions by Track")
    for team in ["left", "right"]:
        team_players = [det for det in players if det.team == team]
        by_track = defaultdict(list)
        for det in team_players:
            by_track[det.track_id].append(det)
        for track_id, dets in by_track.items():
            if len(dets) < 10:
                continue
            mean_x = float(np.mean([det.x for det in dets]))
            mean_y = float(np.mean([det.y for det in dets]))
            rows.append({"track_id": track_id, "team": team, "frames": len(dets), "mean_x": mean_x, "mean_y": mean_y})
            ax.scatter(mean_x, mean_y, s=110, color=colors[team], edgecolor="white", linewidth=1.2)
            ax.text(mean_x + 0.8, mean_y + 0.8, str(track_id), color="white", fontsize=9, weight="bold")
        if team_players:
            cx = float(np.mean([det.x for det in team_players]))
            cy = float(np.mean([det.y for det in team_players]))
            ax.scatter(cx, cy, s=260, marker="*", color=colors[team], edgecolor="black", linewidth=1.0)
            ax.text(cx + 1.2, cy - 2.0, f"{team} centroid", color="white", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return rows


def zone_name(x: float, y: float) -> tuple[str, str]:
    if x < -HALF_LENGTH / 3:
        third = "left_third"
    elif x > HALF_LENGTH / 3:
        third = "right_third"
    else:
        third = "middle_third"
    if y < -HALF_WIDTH / 3:
        channel = "bottom_wing"
    elif y > HALF_WIDTH / 3:
        channel = "top_wing"
    else:
        channel = "central_channel"
    return third, channel


def compute_zone_stats(players: list[Detection], output_csv: Path, output_png: Path) -> list[dict]:
    counts = defaultdict(Counter)
    for det in players:
        if det.team not in {"left", "right"}:
            continue
        third, channel = zone_name(det.x, det.y)
        counts[det.team][third] += 1
        counts[det.team][channel] += 1
    rows = []
    for team, counter in counts.items():
        total = sum(counter[k] for k in ["left_third", "middle_third", "right_third"])
        for zone in ["left_third", "middle_third", "right_third", "bottom_wing", "central_channel", "top_wing"]:
            denom = total if "third" in zone else sum(counter[k] for k in ["bottom_wing", "central_channel", "top_wing"])
            rows.append({"team": team, "zone": zone, "count": counter[zone], "ratio": counter[zone] / denom if denom else 0})
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["team", "zone", "count", "ratio"])
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, zones, title in [
        (axes[0], ["left_third", "middle_third", "right_third"], "Pitch Third Occupancy"),
        (axes[1], ["bottom_wing", "central_channel", "top_wing"], "Channel Occupancy"),
    ]:
        x = np.arange(len(zones))
        width = 0.35
        for idx, team in enumerate(["left", "right"]):
            ratios = [next((row["ratio"] for row in rows if row["team"] == team and row["zone"] == zone), 0) for zone in zones]
            ax.bar(x + (idx - 0.5) * width, ratios, width, label=team)
        ax.set_xticks(x, zones, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("ratio")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return rows


def compute_distances(players: list[Detection], fps: float, output_csv: Path, output_png: Path) -> list[dict]:
    by_track = defaultdict(list)
    for det in players:
        if det.team in {"left", "right"}:
            by_track[det.track_id].append(det)
    rows = []
    for track_id, dets in by_track.items():
        dets = sorted(dets, key=lambda det: det.frame)
        total_dist = 0.0
        valid_steps = 0
        max_speed = 0.0
        for a, b in zip(dets, dets[1:]):
            frame_gap = b.frame - a.frame
            if frame_gap <= 0 or frame_gap > fps * 2:
                continue
            dt = frame_gap / fps
            dist = float(np.hypot(b.x - a.x, b.y - a.y))
            speed = dist / dt if dt else 0.0
            if speed > 12.0:
                continue
            total_dist += dist
            valid_steps += 1
            max_speed = max(max_speed, speed)
        if valid_steps:
            team_votes = Counter(det.team for det in dets if det.team)
            rows.append({
                "track_id": track_id,
                "team": team_votes.most_common(1)[0][0] if team_votes else None,
                "frames": len(dets),
                "distance_m": total_dist,
                "avg_speed_mps": total_dist / (valid_steps / fps) if valid_steps else 0,
                "max_speed_mps": max_speed,
            })
    rows.sort(key=lambda row: row["distance_m"], reverse=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "team", "frames", "distance_m", "avg_speed_mps", "max_speed_mps"])
        writer.writeheader()
        writer.writerows(rows)

    top = rows[:12]
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{row['track_id']} ({row['team']})" for row in top]
    values = [row["distance_m"] for row in top]
    ax.bar(labels, values, color=["#ef4444" if row["team"] == "left" else "#2563eb" for row in top])
    ax.set_title("Top Estimated Running Distance by Track")
    ax.set_ylabel("estimated distance (m)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return rows


def save_centroid_timeline(players: list[Detection], fps: float, output_png: Path, bin_seconds: float = 2.0) -> list[dict]:
    bin_size = int(fps * bin_seconds)
    rows = []
    for team in ["left", "right"]:
        grouped = defaultdict(list)
        for det in players:
            if det.team == team:
                grouped[(det.frame - 1) // bin_size].append(det)
        for idx, dets in grouped.items():
            rows.append({
                "team": team,
                "time_s": idx * bin_seconds,
                "centroid_x": float(np.mean([det.x for det in dets])),
                "centroid_y": float(np.mean([det.y for det in dets])),
                "count": len(dets),
            })
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for team, color in [("left", "#ef4444"), ("right", "#2563eb")]:
        team_rows = [row for row in rows if row["team"] == team]
        xs = [row["time_s"] for row in team_rows]
        axes[0].plot(xs, [row["centroid_x"] for row in team_rows], label=team, color=color)
        axes[1].plot(xs, [row["centroid_y"] for row in team_rows], label=team, color=color)
    axes[0].set_title("Team Centroid Timeline")
    axes[0].set_ylabel("centroid X (m)")
    axes[1].set_ylabel("centroid Y (m)")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.axhline(0, color="#94a3b8", linewidth=1)
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return rows


def save_ball_analysis(detections: list[Detection], fps: float, output_png: Path, output_csv: Path) -> list[dict]:
    balls = sorted([det for det in detections if det.role == "ball"], key=lambda det: det.frame)
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax, "Ball Trajectory and Danger-Zone Entries")
    if balls:
        xs = [det.x for det in balls]
        ys = [det.y for det in balls]
        ax.plot(xs, ys, color="#facc15", linewidth=2.0, alpha=0.85)
        ax.scatter(xs, ys, s=18, color="#facc15", edgecolor="#111827", linewidth=0.2)
        # Penalty-zone danger areas.
        for sign in [-1, 1]:
            ax.axvspan(sign * 36, sign * HALF_LENGTH, color="#ef4444", alpha=0.12)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    danger_frames = []
    for det in balls:
        if abs(det.x) >= 36.0 and abs(det.y) <= 20.16:
            danger_frames.append(det.frame)
    segments = []
    if danger_frames:
        start = prev = danger_frames[0]
        for frame in danger_frames[1:]:
            if frame <= prev + 3:
                prev = frame
            else:
                segments.append((start, prev))
                start = prev = frame
        segments.append((start, prev))
    rows = [{"start_frame": a, "end_frame": b, "start_time_s": a / fps, "end_time_s": b / fps, "duration_s": (b - a + 1) / fps}
            for a, b in segments]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start_frame", "end_frame", "start_time_s", "end_time_s", "duration_s"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_report(output: Path, summary: dict) -> None:
    lines = [
        "# Soccer Tactical Analysis MVP Report",
        "",
        "本报告由 `workflows/tactical/analyze_tactical_metrics.py` 基于 SoccerNetGSR 输出 JSON 自动生成。",
        "",
        "## 已生成分析",
        "",
        "- 球队/全员活动热力图",
        "- 平均站位图",
        "- 区域占比统计",
        "- 球队重心时间曲线",
        "- 球员估算跑动距离与速度",
        "- 足球轨迹和危险区域进入片段",
        "",
        "## 关键统计",
        "",
        f"- 总检测点：`{summary['total_detections']}`",
        f"- 球员检测点：`{summary['player_detections']}`",
        f"- 足球检测点：`{summary['ball_detections']}`",
        f"- 左队球员检测点：`{summary['left_player_detections']}`",
        f"- 右队球员检测点：`{summary['right_player_detections']}`",
        f"- 危险区域片段数：`{summary['danger_segments']}`",
        "",
        "## 注意事项",
        "",
        "- 跑动距离和速度是基于转播画面检测轨迹的估算值，受遮挡、镜头切换、轨迹断裂影响。",
        "- 当前足球轨迹并非每帧都有，因此传球成功率、xG、射门分类暂不建议直接做。",
        "- 当前最稳的下一步是把这些图表嵌入视频片段或做成赛后 HTML/PDF 报告。",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rule-based football tactical analysis outputs.")
    parser.add_argument("--json-path", default="soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-999/SNGS-999.referee_refined.json")
    parser.add_argument("--output-dir", default="soccer_input_dataset/outputs/tactical_analysis")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detections = load_detections(Path(args.json_path))
    players = [det for det in detections if det.role == "player" and det.team in {"left", "right"}]

    save_heatmap(players, output_dir / "all_players_heatmap.png", "All Player Activity Heatmap")
    save_heatmap([det for det in players if det.team == "left"], output_dir / "left_team_heatmap.png", "Left Team Activity Heatmap")
    save_heatmap([det for det in players if det.team == "right"], output_dir / "right_team_heatmap.png", "Right Team Activity Heatmap")
    avg_rows = save_average_positions(players, output_dir / "average_positions.png")
    zone_rows = compute_zone_stats(players, output_dir / "zone_stats.csv", output_dir / "zone_occupancy.png")
    distance_rows = compute_distances(players, args.fps, output_dir / "player_distance_speed.csv", output_dir / "player_distance_top12.png")
    centroid_rows = save_centroid_timeline(players, args.fps, output_dir / "team_centroid_timeline.png")
    danger_rows = save_ball_analysis(detections, args.fps, output_dir / "ball_trajectory_danger.png", output_dir / "danger_zone_segments.csv")

    with open(output_dir / "average_positions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "team", "frames", "mean_x", "mean_y"])
        writer.writeheader()
        writer.writerows(avg_rows)
    with open(output_dir / "team_centroid_timeline.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["team", "time_s", "centroid_x", "centroid_y", "count"])
        writer.writeheader()
        writer.writerows(centroid_rows)

    summary = {
        "total_detections": len(detections),
        "player_detections": len(players),
        "ball_detections": len([det for det in detections if det.role == "ball"]),
        "left_player_detections": len([det for det in players if det.team == "left"]),
        "right_player_detections": len([det for det in players if det.team == "right"]),
        "danger_segments": len(danger_rows),
        "top_distance_tracks": distance_rows[:5],
        "zone_rows": zone_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir / "tactical_analysis_report.md", summary)
    print(f"Wrote tactical analysis outputs to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
