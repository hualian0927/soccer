#!/usr/bin/env python3
"""Create a far-view formation-analysis visualization from SoccerNetGSR JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from make_tactical_visualization_video import (
    COLORS,
    HALF_LENGTH,
    HALF_WIDTH,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PROJECT_ROOT,
    Detection,
    blend_rect,
    current_ball,
    load_by_frame,
    pitch_to_panel,
    put_text,
    role_color,
)


FORMATION_TEMPLATES = {
    "4-3-3": [4, 3, 3],
    "4-2-3-1": [4, 2, 3, 1],
    "4-4-2": [4, 4, 2],
    "3-5-2": [3, 5, 2],
    "3-4-3": [3, 4, 3],
    "4-1-4-1": [4, 1, 4, 1],
    "5-3-2": [5, 3, 2],
    "5-4-1": [5, 4, 1],
}

LINE_LABELS = {
    3: ["后卫线", "中场线", "前锋线"],
    4: ["后卫线", "后腰线", "前腰线", "前锋线"],
}


@dataclass
class FormationLine:
    label: str
    count: int
    center_axis: float
    points: list[tuple[float, float]]
    detections: list[Detection]


@dataclass
class FormationEstimate:
    team: str
    formation: str
    confidence: float
    visible_players: int
    attack_direction: int
    line_counts: list[int]
    scaled_counts: list[int]
    width: float
    depth: float
    compactness: str
    lines: list[FormationLine]
    note: str


class FormationState:
    def __init__(self) -> None:
        self.latest: dict[str, FormationEstimate | None] = {"left": None, "right": None}
        self.formation_counts: dict[str, Counter] = {"left": Counter(), "right": Counter()}
        self.keyframes: list[dict] = []

    def update(self, detections: list[Detection], min_visible: int) -> dict[str, FormationEstimate | None]:
        for team in ("left", "right"):
            estimate = estimate_formation(detections, team, min_visible)
            self.latest[team] = estimate
            if estimate is not None:
                self.formation_counts[team][estimate.formation] += 1
        return self.latest

    def summary(self, frames: int) -> dict:
        return {
            "frames": frames,
            "formation_counts": {team: dict(counter) for team, counter in self.formation_counts.items()},
            "keyframes": self.keyframes,
            "limitations": "阵型为单目转播画面和GSR场地坐标上的可见球员近似估计；镜头只覆盖局部区域、检测漏检或球队换位会降低可信度。",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export formation-analysis mp4 from SoccerNetGSR JSON.")
    parser.add_argument("--input-video", default="soccer_input_dataset/test_5min.mp4")
    parser.add_argument("--json-path", default="soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-1000/SNGS-1000.json")
    parser.add_argument("--output-video", default="soccer_input_dataset/outputs/test_5min_formation_analysis.mp4")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="Use 0 for the full input video.")
    parser.add_argument("--snapshot-frame", type=int, default=4500)
    parser.add_argument("--min-visible", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=52.0)
    parser.add_argument("--keyframe-pause-sec", type=float, default=1.2)
    parser.add_argument("--keyframe-cooldown-sec", type=float, default=18.0)
    parser.add_argument("--max-keyframes", type=int, default=10)
    return parser.parse_args()


def zh_team(team: str) -> str:
    return {"left": "蓝队", "right": "白队"}.get(team, team)


def player_detections(detections: list[Detection], team: str, include_goalkeeper: bool = False) -> list[Detection]:
    roles = {"player", "goalkeeper"} if include_goalkeeper else {"player"}
    players = [
        det
        for det in detections
        if det.team == team
        and det.role in roles
        and det.pitch_x is not None
        and det.pitch_y is not None
        and -HALF_LENGTH <= float(det.pitch_x) <= HALF_LENGTH
        and -HALF_WIDTH <= float(det.pitch_y) <= HALF_WIDTH
    ]
    best_by_track: dict[int, Detection] = {}
    for det in players:
        old = best_by_track.get(det.track_id)
        if old is None or bbox_area(det) > bbox_area(old):
            best_by_track[det.track_id] = det
    unique_players = list(best_by_track.values())
    unique_players.sort(key=lambda det: (det.role != "goalkeeper", bbox_area(det)), reverse=True)
    return unique_players[:10] if len(unique_players) > 10 else unique_players


def bbox_area(det: Detection) -> float:
    if not det.bbox:
        return 0.0
    return max(0.0, float(det.bbox.get("w", 0.0))) * max(0.0, float(det.bbox.get("h", 0.0)))


def infer_attack_direction(detections: list[Detection], team: str) -> int:
    goalkeepers = [
        det
        for det in detections
        if det.team == team and det.role == "goalkeeper" and det.pitch_x is not None
    ]
    if goalkeepers:
        gk_x = float(np.mean([float(det.pitch_x) for det in goalkeepers]))
        return 1 if gk_x < 0 else -1
    own = player_detections(detections, team, include_goalkeeper=False)
    opp = player_detections(detections, "right" if team == "left" else "left", include_goalkeeper=False)
    if own and opp:
        own_x = float(np.mean([float(det.pitch_x) for det in own]))
        opp_x = float(np.mean([float(det.pitch_x) for det in opp]))
        return 1 if own_x < opp_x else -1
    return 1 if team == "left" else -1


def kmeans_1d(values: np.ndarray, k: int, iterations: int = 16) -> tuple[np.ndarray, np.ndarray, float]:
    if len(values) < k:
        labels = np.arange(len(values), dtype=np.int32)
        centers = values.copy()
        return labels, centers, 0.0
    quantiles = np.linspace(0.08, 0.92, k)
    centers = np.quantile(values, quantiles).astype(np.float32)
    labels = np.zeros(len(values), dtype=np.int32)
    for _ in range(iterations):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1).astype(np.int32)
        new_centers = centers.copy()
        for idx in range(k):
            cluster_values = values[labels == idx]
            if len(cluster_values):
                new_centers[idx] = float(cluster_values.mean())
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    order = np.argsort(centers)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    ordered_centers = centers[order]
    ordered_labels = np.array([remap[int(label)] for label in labels], dtype=np.int32)
    sse = float(np.mean((values - ordered_centers[ordered_labels]) ** 2)) if len(values) else 0.0
    return ordered_labels, ordered_centers, sse


def scaled_line_counts(counts: list[int], total: int = 10) -> list[int]:
    raw = np.array(counts, dtype=np.float32)
    if raw.sum() <= 0:
        return counts
    scaled = raw / raw.sum() * total
    floors = np.floor(scaled).astype(int)
    remainder = total - int(floors.sum())
    order = np.argsort(-(scaled - floors))
    for idx in order[:remainder]:
        floors[idx] += 1
    floors = np.maximum(floors, 1)
    while int(floors.sum()) > total:
        idx = int(np.argmax(floors))
        if floors[idx] <= 1:
            break
        floors[idx] -= 1
    return [int(v) for v in floors.tolist()]


def template_distance(counts: list[int], template: list[int]) -> float:
    if len(counts) != len(template):
        return 999.0
    a = np.array(counts, dtype=np.float32)
    b = np.array(template, dtype=np.float32)
    a = a / max(float(a.sum()), 1.0)
    b = b / max(float(b.sum()), 1.0)
    return float(np.abs(a - b).sum())


def classify_formation(line_counts: list[int], visible_players: int, depth: float, width: float) -> tuple[str, float, list[int], str]:
    scaled = scaled_line_counts(line_counts, 10)
    candidates = [(name, tpl) for name, tpl in FORMATION_TEMPLATES.items() if len(tpl) == len(line_counts)]
    if not candidates:
        return "局部阵型", 0.0, scaled, "可见人数不足，暂不稳定判定"
    name, template = min(candidates, key=lambda item: template_distance(scaled, item[1]))
    dist = template_distance(scaled, template)
    visible_factor = min(1.0, visible_players / 10.0)
    shape_factor = min(1.0, max(0.35, depth / 45.0) * 0.65 + max(0.35, width / 42.0) * 0.35)
    confidence = max(0.0, min(100.0, (1.0 - dist / 0.95) * 100.0 * visible_factor * shape_factor))
    note = f"可见线型 {line_counts}，按10人缩放为 {scaled}，匹配{name}"
    if visible_players < 9:
        note += "；当前为局部镜头，置信度下调"
    return name, confidence, scaled, note


def compactness_label(width: float, depth: float) -> str:
    area_proxy = width * depth
    if area_proxy < 850:
        return "紧凑"
    if area_proxy < 1550:
        return "均衡"
    return "拉开"


def estimate_formation(detections: list[Detection], team: str, min_visible: int) -> FormationEstimate | None:
    players = player_detections(detections, team, include_goalkeeper=False)
    if len(players) < min_visible:
        return None
    direction = infer_attack_direction(detections, team)
    axes = np.array([float(det.pitch_x) * direction for det in players], dtype=np.float32)
    xs = np.array([float(det.pitch_x) for det in players], dtype=np.float32)
    ys = np.array([float(det.pitch_y) for det in players], dtype=np.float32)
    depth = float(xs.max() - xs.min()) if len(xs) > 1 else 0.0
    width = float(ys.max() - ys.min()) if len(ys) > 1 else 0.0

    best: tuple[int, np.ndarray, np.ndarray, float] | None = None
    for k in (3, 4):
        if len(players) < k + 3:
            continue
        labels, centers, sse = kmeans_1d(axes, k)
        empty_penalty = sum(1 for idx in range(k) if int(np.sum(labels == idx)) == 0) * 1000.0
        score = sse + empty_penalty + (25.0 if k == 4 and len(players) < 9 else 0.0)
        if best is None or score < best[3]:
            best = (k, labels, centers, score)
    if best is None:
        return None

    k, labels, centers, _ = best
    line_labels = LINE_LABELS[k]
    lines: list[FormationLine] = []
    line_counts: list[int] = []
    for idx in range(k):
        group_dets = [det for det, label in zip(players, labels) if int(label) == idx]
        group_points = [(float(det.pitch_x), float(det.pitch_y)) for det in group_dets]
        line_counts.append(len(group_dets))
        lines.append(FormationLine(line_labels[idx], len(group_dets), float(centers[idx]), group_points, group_dets))

    formation, confidence, scaled, note = classify_formation(line_counts, len(players), depth, width)
    if confidence < 36.0:
        formation = f"{formation}?"
    return FormationEstimate(
        team=team,
        formation=formation,
        confidence=confidence,
        visible_players=len(players),
        attack_direction=direction,
        line_counts=line_counts,
        scaled_counts=scaled,
        width=width,
        depth=depth,
        compactness=compactness_label(width, depth),
        lines=lines,
        note=note,
    )


def draw_formation_on_pitch(
    frame: np.ndarray,
    estimate: FormationEstimate | None,
    origin: tuple[int, int],
    size: tuple[int, int],
    show_summary: bool = True,
) -> None:
    if estimate is None:
        return
    color = COLORS[estimate.team]
    point_radius = 4 if size[0] < 260 else 5
    label_scale = 0.28 if size[0] < 260 else 0.34
    for line in estimate.lines:
        if not line.points:
            continue
        pts = sorted(line.points, key=lambda p: p[1])
        panel_points = [pitch_to_panel(x, y, origin, size) for x, y in pts]
        if len(panel_points) >= 2:
            cv2.polylines(frame, [np.array(panel_points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        for px, py in panel_points:
            cv2.circle(frame, (px, py), point_radius, color, 1, cv2.LINE_AA)
        label_x = int(np.mean([p[0] for p in panel_points]))
        label_y = int(np.mean([p[1] for p in panel_points]))
        put_text(frame, f"{line.label}{line.count}", (label_x + 4, label_y - 4), scale=label_scale, color=color)
    if show_summary:
        ox, oy = origin
        put_text(frame, f"{zh_team(estimate.team)} {estimate.formation} {estimate.confidence:.0f}%", (ox, oy + size[1] + (18 if estimate.team == "left" else 40)), scale=0.42, color=color)


def draw_clean_pitch_base(frame: np.ndarray, origin: tuple[int, int], size: tuple[int, int]) -> None:
    ox, oy = origin
    width, height = size
    cv2.rectangle(frame, (ox, oy), (ox + width, oy + height), COLORS["pitch"], -1)
    line = COLORS["white"]
    cv2.rectangle(frame, (ox, oy), (ox + width, oy + height), line, 1)
    cv2.line(frame, (ox + width // 2, oy), (ox + width // 2, oy + height), line, 1)
    cv2.circle(frame, (ox + width // 2, oy + height // 2), max(7, int(height * 9.15 / PITCH_WIDTH)), line, 1)
    cv2.circle(frame, (ox + width // 2, oy + height // 2), 2, line, -1)

    pa_w = int(width * 16.5 / PITCH_LENGTH)
    pa_h = int(height * 40.32 / PITCH_WIDTH)
    six_w = int(width * 5.5 / PITCH_LENGTH)
    six_h = int(height * 18.32 / PITCH_WIDTH)
    for left_side in [True, False]:
        gx = ox if left_side else ox + width
        if left_side:
            cv2.rectangle(frame, (gx, oy + (height - pa_h) // 2), (gx + pa_w, oy + (height + pa_h) // 2), line, 1)
            cv2.rectangle(frame, (gx, oy + (height - six_h) // 2), (gx + six_w, oy + (height + six_h) // 2), line, 1)
        else:
            cv2.rectangle(frame, (gx - pa_w, oy + (height - pa_h) // 2), (gx, oy + (height + pa_h) // 2), line, 1)
            cv2.rectangle(frame, (gx - six_w, oy + (height - six_h) // 2), (gx, oy + (height + six_h) // 2), line, 1)


def clean_board_text(state: FormationState) -> tuple[str, str]:
    estimates = [est for est in state.latest.values() if est is not None]
    if not estimates:
        return "阵型板", "站位不足"
    primary = max(estimates, key=lambda est: est.confidence)
    counts = "-".join(str(v) for v in primary.scaled_counts)
    return "阵型板", f"{zh_team(primary.team)} {primary.formation} 线型{counts}"


def draw_clean_formation_board(
    frame: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    detections: list[Detection],
    state: FormationState,
) -> None:
    ox, oy = origin
    width, height = size
    board_h = height + 52
    blend_rect(frame, ox - 8, oy - 34, ox + width + 8, oy + board_h, COLORS["panel"], 0.76)
    title, subtitle = clean_board_text(state)
    put_text(frame, title, (ox, oy - 12), scale=0.46, thickness=1)
    put_text(frame, subtitle, (ox + 58, oy - 12), scale=0.34, color=(220, 225, 230))

    draw_clean_pitch_base(frame, origin, size)
    for estimate in state.latest.values():
        draw_formation_on_pitch(frame, estimate, origin, size, show_summary=False)

    ball = current_ball(detections)
    if ball is not None and ball.pitch_x is not None and ball.pitch_y is not None:
        px, py = pitch_to_panel(float(ball.pitch_x), float(ball.pitch_y), origin, size)
        cv2.circle(frame, (px, py), 3, COLORS["ball"], -1, cv2.LINE_AA)

    legend_y = oy + height + 20
    for idx, team in enumerate(("left", "right")):
        est = state.latest.get(team)
        x = ox + idx * max(92, width // 2)
        cv2.circle(frame, (x + 6, legend_y - 5), 4, COLORS[team], -1, cv2.LINE_AA)
        text = f"{zh_team(team)} {est.formation if est else '--'}"
        put_text(frame, text, (x + 16, legend_y), scale=0.30, color=COLORS[team])


def bbox_center(det: Detection) -> tuple[int, int] | None:
    if not det.bbox:
        return None
    x = float(det.bbox.get("x", 0))
    y = float(det.bbox.get("y", 0))
    w = float(det.bbox.get("w", 0))
    h = float(det.bbox.get("h", 0))
    if w <= 0 or h <= 0:
        return None
    return int(x + w / 2), int(y + h * 0.55)


def draw_formation_on_frame(frame: np.ndarray, estimate: FormationEstimate | None) -> None:
    if estimate is None:
        return
    color = COLORS[estimate.team]
    for line in estimate.lines:
        centers = [bbox_center(det) for det in line.detections]
        centers = [pt for pt in centers if pt is not None]
        if len(centers) >= 2:
            centers = sorted(centers, key=lambda p: p[0])
            cv2.polylines(frame, [np.array(centers, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        if centers:
            lx = int(np.mean([p[0] for p in centers]))
            ly = int(np.mean([p[1] for p in centers]))
            put_text(frame, line.label, (lx + 4, max(18, ly - 8)), scale=0.38, color=color)


def draw_detection_boxes(frame: np.ndarray, detections: list[Detection]) -> None:
    for det in detections:
        if not det.bbox:
            continue
        x = int(det.bbox.get("x", 0))
        y = int(det.bbox.get("y", 0))
        w = int(det.bbox.get("w", 0))
        h = int(det.bbox.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        color = role_color(det)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2 if det.role != "ball" else 3)
        if det.role == "ball":
            label = "球"
        elif det.role == "referee":
            label = f"裁判 {det.track_id}"
        elif det.team in {"left", "right"}:
            label = f"{zh_team(det.team)} {det.track_id}"
        else:
            label = f"ID {det.track_id}"
        put_text(frame, label, (x, max(16, y - 6)), scale=0.38, color=color)


def formation_panel_lines(state: FormationState) -> tuple[str, list[str]]:
    estimates = [state.latest.get("left"), state.latest.get("right")]
    good = [est for est in estimates if est is not None]
    if not good:
        return "阵型暂不稳定：可见人数不足", ["等待远景或更多完整站位进入画面"]
    primary = max(good, key=lambda est: est.confidence)
    title = f"{zh_team(primary.team)}阵型倾向：{primary.formation}（{primary.confidence:.0f}%）"
    lines = []
    for team in ("left", "right"):
        est = state.latest.get(team)
        if est is None:
            lines.append(f"{zh_team(team)}：可见人数不足，暂不判定")
        else:
            lines.append(
                f"{zh_team(team)}：{est.formation}  线型{est.line_counts}->约{est.scaled_counts}  宽{est.width:.1f}m 深{est.depth:.1f}m {est.compactness}"
            )
    lines.append(primary.note)
    return title, lines


def draw_formation_panel(frame: np.ndarray, frame_idx: int, fps: float, state: FormationState) -> None:
    title, lines = formation_panel_lines(state)
    x1, y1, x2, y2 = 20, frame.shape[0] - 238, 780, frame.shape[0] - 18
    blend_rect(frame, x1, y1, x2, y2, COLORS["panel"], 0.72)
    put_text(frame, "阵型分析", (x1 + 14, y1 + 30), scale=0.68, thickness=2)
    put_text(frame, title, (x1 + 14, y1 + 64), scale=0.54, thickness=1)
    put_text(frame, f"时间 {frame_idx / fps:05.2f}s  帧 {frame_idx}", (x1 + 14, y1 + 94), scale=0.42)
    for idx, line in enumerate(lines[:4]):
        put_text(frame, line, (x1 + 14, y1 + 124 + idx * 27), scale=0.40)


def keyframe_candidate(state: FormationState, threshold: float) -> FormationEstimate | None:
    estimates = [est for est in state.latest.values() if est is not None and est.confidence >= threshold]
    if not estimates:
        return None
    return max(estimates, key=lambda est: est.confidence)


def draw_keyframe_overlay(frame: np.ndarray, frame_idx: int, fps: float, state: FormationState, reason: FormationEstimate) -> None:
    x1, y1, x2, y2 = 300, 54, frame.shape[1] - 90, 282
    blend_rect(frame, x1, y1, x2, y2, COLORS["panel"], 0.82)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[reason.team], 2)
    put_text(frame, "阵型关键帧暂停", (x1 + 18, y1 + 40), scale=0.72, thickness=2, color=COLORS[reason.team])
    title, lines = formation_panel_lines(state)
    put_text(frame, title, (x1 + 18, y1 + 78), scale=0.52)
    put_text(frame, f"原因：可见{reason.visible_players}名球员，纵深线型稳定，适合观察{reason.formation}结构", (x1 + 18, y1 + 110), scale=0.42)
    for idx, line in enumerate(lines[:3]):
        put_text(frame, line, (x1 + 18, y1 + 142 + idx * 26), scale=0.38)
    put_text(frame, "解读：阵型编号是对可见球员站位的近似匹配，镜头局部时以线型趋势为主", (x1 + 18, y2 - 18), scale=0.36, color=(220, 225, 230))


def draw_frame(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    detections: list[Detection],
    state: FormationState,
    min_visible: int,
) -> np.ndarray:
    output = frame.copy()
    state.update(detections, min_visible)
    draw_detection_boxes(output, detections)
    for estimate in state.latest.values():
        draw_formation_on_frame(output, estimate)

    pitch_w = max(170, output.shape[1] // 6)
    pitch_h = int(pitch_w * PITCH_WIDTH / PITCH_LENGTH)
    origin = (output.shape[1] - pitch_w - 22, 46)
    draw_clean_formation_board(output, origin, (pitch_w, pitch_h), detections, state)
    draw_formation_panel(output, frame_idx, fps, state)
    return output


def main() -> int:
    args = parse_args()
    input_video = (PROJECT_ROOT / args.input_video).resolve()
    json_path = (PROJECT_ROOT / args.json_path).resolve()
    output_video = (PROJECT_ROOT / args.output_video).resolve()
    report_path = output_video.with_suffix(".formation_report.json")
    snapshot_path = output_video.with_name(f"{output_video.stem}_frame{args.snapshot_frame:03d}.jpg")

    by_frame = load_by_frame(json_path)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(1, args.start_frame)
    if start_frame > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    remaining_frames = max(0, total_frames - start_frame + 1)
    frame_limit = remaining_frames if args.max_frames <= 0 else min(args.max_frames, remaining_frames)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")

    state = FormationState()
    pause_frames = max(0, int(args.keyframe_pause_sec * fps))
    cooldown_frames = max(1, int(args.keyframe_cooldown_sec * fps))
    last_keyframe_frame = -10**9
    last_keyframe_signature = ""
    inserted_keyframes = 0
    wrote_snapshot = False
    written_frames = 0

    for local_idx in range(1, frame_limit + 1):
        ok, frame = cap.read()
        if not ok:
            break
        source_frame = start_frame + local_idx - 1
        detections = by_frame.get(source_frame, [])
        visualized = draw_frame(frame, source_frame, fps, detections, state, args.min_visible)
        writer.write(visualized)
        written_frames += 1
        if local_idx == args.snapshot_frame:
            cv2.imwrite(str(snapshot_path), visualized)
            wrote_snapshot = True

        candidate = keyframe_candidate(state, args.confidence_threshold)
        signature = f"{candidate.team}:{candidate.formation}" if candidate else ""
        if (
            candidate is not None
            and inserted_keyframes < args.max_keyframes
            and source_frame - last_keyframe_frame >= cooldown_frames
            and (signature != last_keyframe_signature or candidate.confidence >= args.confidence_threshold + 16.0)
        ):
            pause = visualized.copy()
            draw_keyframe_overlay(pause, source_frame, fps, state, candidate)
            for _ in range(pause_frames):
                writer.write(pause)
                written_frames += 1
            state.keyframes.append(
                {
                    "frame": source_frame,
                    "time_sec": source_frame / fps,
                    "team": candidate.team,
                    "formation": candidate.formation,
                    "confidence": candidate.confidence,
                    "visible_players": candidate.visible_players,
                    "line_counts": candidate.line_counts,
                    "scaled_counts": candidate.scaled_counts,
                }
            )
            last_keyframe_frame = source_frame
            last_keyframe_signature = signature
            inserted_keyframes += 1

    cap.release()
    writer.release()
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(state.summary(local_idx if "local_idx" in locals() else 0), f, ensure_ascii=False, indent=2)
    print(f"Wrote {written_frames} frames at {fps:.2f} fps to {output_video}")
    print(f"Inserted {inserted_keyframes} formation keyframe pauses")
    print(f"Wrote formation report to {report_path}")
    if wrote_snapshot:
        print(f"Wrote snapshot to {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
