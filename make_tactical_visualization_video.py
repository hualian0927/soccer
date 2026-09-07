#!/usr/bin/env python3
"""Create a tactical-analysis visualization video from local SoccerNetGSR output."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
CHINESE_FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2
HALF_WIDTH = PITCH_WIDTH / 2


COLORS = {
    "left": (235, 105, 35),
    "right": (245, 245, 245),
    "referee": (235, 40, 235),
    "ball": (20, 220, 255),
    "other": (20, 165, 255),
    "white": (245, 245, 245),
    "black": (20, 24, 28),
    "panel": (18, 22, 26),
    "pitch": (43, 126, 49),
    "danger": (40, 45, 245),
}

DISPLAY_COLORS = {
    "blue": (235, 105, 35),
    "white": (245, 245, 245),
    "red": (45, 45, 235),
    "yellow": (20, 220, 255),
    "green": (55, 200, 75),
}
TEAM_LABELS = {"left": "蓝队", "right": "白队"}


@dataclass
class Detection:
    frame: int
    track_id: int
    role: str
    team: str | None
    bbox: dict | None
    pitch_x: float | None
    pitch_y: float | None


class TacticalState:
    """Rolling tactical metrics for the current rendered clip."""

    def __init__(self, heat_cols: int = 21, heat_rows: int = 14):
        self.possession = Counter()
        self.thirds = Counter()
        self.channels = Counter()
        self.heat_cols = heat_cols
        self.heat_rows = heat_rows
        self.heatmaps = {
            "left": np.zeros((heat_rows, heat_cols), dtype=np.float32),
            "right": np.zeros((heat_rows, heat_cols), dtype=np.float32),
        }
        self.latest_possession = "unknown"
        self.latest_third = "unknown"
        self.latest_channel = "unknown"
        self.latest_shapes: dict[str, dict[str, float]] = {}
        self.last_stable_possession: str | None = None
        self.last_ball_position: tuple[float, float] | None = None
        self.transitions = 0
        self.latest_transition = "none"
        self.danger_attacks = 0
        self.latest_danger = "none"
        self.pressure = Counter()
        self.latest_pressure = "unknown"
        self.latest_overload = "unknown"
        self.latest_compactness: dict[str, dict[str, float | str]] = {}
        self.ball_speed = 0.0
        self.latest_ball_position: tuple[float, float] | None = None

    def _heat_bin(self, x: float, y: float) -> tuple[int, int] | None:
        if not (-HALF_LENGTH <= x <= HALF_LENGTH and -HALF_WIDTH <= y <= HALF_WIDTH):
            return None
        col = int((x + HALF_LENGTH) / PITCH_LENGTH * self.heat_cols)
        row = int((HALF_WIDTH - y) / PITCH_WIDTH * self.heat_rows)
        col = max(0, min(self.heat_cols - 1, col))
        row = max(0, min(self.heat_rows - 1, row))
        return row, col

    def update(self, detections: list[Detection], ball: Detection | None, fps: float) -> None:
        possession = possession_team(detections, ball)
        third, channel, _ = ball_zone(ball)
        self.latest_possession = possession
        self.latest_third = third
        self.latest_channel = channel
        self.possession[possession] += 1
        self.thirds[third] += 1
        self.channels[channel] += 1
        self.latest_shapes = team_shapes(detections)
        self.latest_compactness = team_compactness(detections)

        if ball is not None and ball.pitch_x is not None and ball.pitch_y is not None:
            current_ball = (float(ball.pitch_x), float(ball.pitch_y))
            if self.last_ball_position is not None:
                dist = float(np.hypot(current_ball[0] - self.last_ball_position[0], current_ball[1] - self.last_ball_position[1]))
                self.ball_speed = dist * max(fps, 1.0)
            self.last_ball_position = current_ball
            self.latest_ball_position = current_ball

        if possession in {"left", "right"}:
            if self.last_stable_possession in {"left", "right"} and possession != self.last_stable_possession:
                self.transitions += 1
                self.latest_transition = f"{self.last_stable_possession}->{possession}"
            self.last_stable_possession = possession

        danger = danger_attack(detections, ball, possession)
        self.latest_danger = danger["label"]
        if danger["active"]:
            self.danger_attacks += 1

        pressure = pressure_state(detections, ball, possession)
        self.latest_pressure = pressure["label"]
        self.pressure[pressure["label"]] += 1
        self.latest_overload = overload_state(detections, ball)["label"]

        for det in detections:
            if det.team not in {"left", "right"} or det.role not in {"player", "goalkeeper"}:
                continue
            if det.pitch_x is None or det.pitch_y is None:
                continue
            bin_pos = self._heat_bin(det.pitch_x, det.pitch_y)
            if bin_pos is not None:
                self.heatmaps[det.team][bin_pos] += 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a tactical visualization mp4 from SoccerNetGSR JSON.")
    parser.add_argument("--input-video", default="soccer_input_dataset/test.mp4")
    parser.add_argument(
        "--json-path",
        default="soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-999/SNGS-999.referee_refined.json",
    )
    parser.add_argument(
        "--output-video",
        default="soccer_input_dataset/outputs/test_tactical_visualized.mp4",
    )
    parser.add_argument("--snapshot-frame", type=int, default=301)
    parser.add_argument("--trail-frames", type=int, default=80)
    parser.add_argument("--start-frame", type=int, default=1, help="1-based frame offset in the input video.")
    parser.add_argument("--max-frames", type=int, default=0, help="Use 0 for the full input video.")
    parser.add_argument("--yolo-fallback", action="store_true", help="Draw extra close-up boxes when GSR has few boxes.")
    parser.add_argument("--fallback-backend", choices=["yolox", "ultralytics"], default="yolox")
    parser.add_argument("--yolox-exp", default="exp/yolox_x_soccernet.py")
    parser.add_argument("--yolox-ckpt", default="checkpoints/yolox_soccernet.pth.tar")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--fallback-min-boxes", type=int, default=4)
    parser.add_argument("--fallback-conf", type=float, default=0.25)
    parser.add_argument("--team0-label", default="蓝队", help="Display label for JSON team 'left'.")
    parser.add_argument("--team1-label", default="白队", help="Display label for JSON team 'right'.")
    parser.add_argument("--team0-display-color", choices=sorted(DISPLAY_COLORS), default="blue")
    parser.add_argument("--team1-display-color", choices=sorted(DISPLAY_COLORS), default="white")
    return parser.parse_args()


def parse_frame(image_id: str) -> int:
    return int(str(image_id)[-6:])


def normalize_team(value) -> str | None:
    if value is None:
        return None
    text = str(value).lower()
    if text in {"left", "right"}:
        return text
    return None


def load_by_frame(json_path: Path) -> dict[int, list[Detection]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for pred in data.get("predictions", []):
        attrs = pred.get("attributes") or {}
        pitch = pred.get("bbox_pitch") or {}
        pitch_x = pitch.get("x_bottom_middle")
        pitch_y = pitch.get("y_bottom_middle")
        if pitch_x is not None:
            pitch_x = float(pitch_x)
        if pitch_y is not None:
            pitch_y = float(pitch_y)
        det = Detection(
            frame=parse_frame(pred.get("image_id", "0")),
            track_id=int(pred.get("track_id", -1)),
            role=str(attrs.get("role", "")).lower(),
            team=normalize_team(attrs.get("team")),
            bbox=pred.get("bbox_image"),
            pitch_x=pitch_x,
            pitch_y=pitch_y,
        )
        by_frame[det.frame].append(det)
    return by_frame


def role_color(det: Detection) -> tuple[int, int, int]:
    if det.role == "ball":
        return COLORS["ball"]
    if det.role == "referee":
        return COLORS["referee"]
    if det.role in {"fallback_person", "fallback_ball"}:
        return COLORS["other"] if det.role == "fallback_person" else COLORS["ball"]
    if det.team in {"left", "right"}:
        return COLORS[det.team]
    return COLORS["other"]


def bbox_iou(box_a: dict, box_b: dict) -> float:
    ax1 = float(box_a.get("x", 0))
    ay1 = float(box_a.get("y", 0))
    ax2 = ax1 + float(box_a.get("w", 0))
    ay2 = ay1 + float(box_a.get("h", 0))
    bx1 = float(box_b.get("x", 0))
    by1 = float(box_b.get("y", 0))
    bx2 = bx1 + float(box_b.get("w", 0))
    by2 = by1 + float(box_b.get("h", 0))
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ultralytics_model(model_name: str):
    try:
        from ultralytics import YOLO
        return YOLO(model_name)
    except Exception as exc:
        print(f"Ultralytics fallback disabled: {exc}")
        return None


def load_yolox_model(exp_file: Path, ckpt_file: Path, conf: float):
    try:
        from inference_soccernetGSR import Predictor
        from yolox.exp import get_exp

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        exp = get_exp(str(exp_file), "yolo-x")
        exp.test_conf = conf
        model = exp.get_model().to(device)
        model.eval()
        ckpt = torch.load(str(ckpt_file), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        return Predictor(model=model, exp=exp, device=device, fp16=False)
    except Exception as exc:
        print(f"YOLOX fallback disabled: {exc}")
        return None


def load_fallback_model(args: argparse.Namespace):
    if not args.yolo_fallback:
        return None
    if args.fallback_backend == "ultralytics":
        return load_ultralytics_model(args.yolo_model)
    return load_yolox_model(
        (PROJECT_ROOT / args.yolox_exp).resolve(),
        (PROJECT_ROOT / args.yolox_ckpt).resolve(),
        args.fallback_conf,
    )


def ultralytics_fallback_detections(
    model,
    frame: np.ndarray,
    existing: list[Detection],
    conf: float,
) -> list[Detection]:
    if model is None:
        return []
    result = model.predict(frame, imgsz=960, conf=conf, classes=[0, 32], verbose=False)[0]
    existing_boxes = [det.bbox for det in existing if det.bbox]
    fallback: list[Detection] = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        bbox = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        if any(bbox_iou(bbox, old_box) > 0.35 for old_box in existing_boxes):
            continue
        role = "fallback_ball" if cls_id == 32 else "fallback_person"
        fallback.append(
            Detection(
                frame=-1,
                track_id=-1,
                role=role,
                team=None,
                bbox=bbox,
                pitch_x=None,
                pitch_y=None,
            )
        )
    return fallback


def yolox_fallback_detections(
    predictor,
    frame: np.ndarray,
    existing: list[Detection],
) -> list[Detection]:
    if predictor is None:
        return []
    from yolox.tracking_utils.timer import Timer

    outputs, img_info = predictor.inference(frame, Timer())
    output = outputs[0] if outputs else None
    if output is None or len(output) == 0:
        return []

    height, width = frame.shape[:2]
    dets = output.cpu().detach().numpy()
    scale = min(predictor.test_size[1] / width, predictor.test_size[0] / height)
    dets[:, :4] /= scale
    existing_boxes = [det.bbox for det in existing if det.bbox]
    fallback: list[Detection] = []
    for x1, y1, x2, y2, *_ in dets:
        x1 = float(max(0, min(width - 1, x1)))
        y1 = float(max(0, min(height - 1, y1)))
        x2 = float(max(0, min(width - 1, x2)))
        y2 = float(max(0, min(height - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        bbox = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        if any(bbox_iou(bbox, old_box) > 0.35 for old_box in existing_boxes):
            continue
        fallback.append(
            Detection(
                frame=-1,
                track_id=-1,
                role="fallback_person",
                team=None,
                bbox=bbox,
                pitch_x=None,
                pitch_y=None,
            )
        )
    return fallback


def fallback_detections(
    model,
    backend: str,
    frame: np.ndarray,
    existing: list[Detection],
    conf: float,
) -> list[Detection]:
    if backend == "ultralytics":
        return ultralytics_fallback_detections(model, frame, existing, conf)
    return yolox_fallback_detections(model, frame, existing)


_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size not in _FONT_CACHE:
        if CHINESE_FONT_PATH.exists():
            _FONT_CACHE[size] = ImageFont.truetype(str(CHINESE_FONT_PATH), size=size)
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.52,
    color: tuple[int, int, int] = COLORS["white"],
    thickness: int = 1,
) -> None:
    if has_cjk(text):
        font_size = max(10, int(scale * 38))
        font = load_font(font_size)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        x, baseline_y = origin
        y = max(0, baseline_y - font_size)
        rgb_color = (int(color[2]), int(color[1]), int(color[0]))
        draw.text((x, y), text, font=font, fill=rgb_color)
        image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        return
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def blend_rect(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    x1 = max(0, min(image.shape[1], x1))
    x2 = max(0, min(image.shape[1], x2))
    y1 = max(0, min(image.shape[0], y1))
    y2 = max(0, min(image.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def pitch_to_panel(x: float, y: float, origin: tuple[int, int], size: tuple[int, int]) -> tuple[int, int]:
    ox, oy = origin
    width, height = size
    px = ox + int((x + HALF_LENGTH) / PITCH_LENGTH * width)
    py = oy + int((HALF_WIDTH - y) / PITCH_WIDTH * height)
    return px, py


def draw_pitch_guides(frame: np.ndarray, origin: tuple[int, int], size: tuple[int, int]) -> None:
    ox, oy = origin
    width, height = size
    guide = (220, 220, 220)
    for x_frac in (1 / 3, 2 / 3):
        x = ox + int(width * x_frac)
        cv2.line(frame, (x, oy), (x, oy + height), guide, 1, cv2.LINE_AA)
    for y_frac in (1 / 3, 2 / 3):
        y = oy + int(height * y_frac)
        cv2.line(frame, (ox, y), (ox + width, y), guide, 1, cv2.LINE_AA)


def draw_heatmap(frame: np.ndarray, origin: tuple[int, int], size: tuple[int, int], state: TacticalState | None) -> None:
    if state is None:
        return
    ox, oy = origin
    width, height = size
    overlay = frame.copy()
    for team in ("left", "right"):
        heat = state.heatmaps[team]
        max_value = float(heat.max())
        if max_value <= 0:
            continue
        color = COLORS[team]
        cell_w = width / state.heat_cols
        cell_h = height / state.heat_rows
        for row in range(state.heat_rows):
            for col in range(state.heat_cols):
                value = float(heat[row, col])
                if value <= 0:
                    continue
                alpha_scale = min(1.0, value / max_value)
                radius = max(2, int(min(cell_w, cell_h) * (0.30 + 0.45 * alpha_scale)))
                cx = ox + int((col + 0.5) * cell_w)
                cy = oy + int((row + 0.5) * cell_h)
                cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)


def draw_analysis_zones(frame: np.ndarray, origin: tuple[int, int], size: tuple[int, int], state: TacticalState | None) -> None:
    if state is None or state.latest_ball_position is None:
        return
    bx, by = state.latest_ball_position
    px, py = pitch_to_panel(bx, by, origin, size)
    width, height = size
    radius_8 = max(2, int(8.0 / PITCH_LENGTH * width))
    radius_18 = max(radius_8 + 1, int(18.0 / PITCH_LENGTH * width))
    cv2.circle(frame, (px, py), radius_18, (120, 210, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (px, py), radius_8, COLORS["danger"], 1, cv2.LINE_AA)
    if state.latest_danger not in {"normal", "no ball", "final third"}:
        cv2.circle(frame, (px, py), radius_18 + 2, COLORS["danger"], 2, cv2.LINE_AA)


def draw_mini_pitch(
    frame: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    detections: list[Detection],
    ball_trail: deque[tuple[float, float]],
    state: TacticalState | None = None,
) -> None:
    ox, oy = origin
    width, height = size
    blend_rect(frame, ox - 10, oy - 34, ox + width + 10, oy + height + 34, COLORS["panel"], 0.68)
    put_text(frame, "球员分布", (ox, oy - 12), scale=0.58, thickness=1)

    cv2.rectangle(frame, (ox, oy), (ox + width, oy + height), COLORS["pitch"], -1)
    line = COLORS["white"]
    cv2.rectangle(frame, (ox, oy), (ox + width, oy + height), line, 1)
    cv2.line(frame, (ox + width // 2, oy), (ox + width // 2, oy + height), line, 1)
    cv2.circle(frame, (ox + width // 2, oy + height // 2), int(height * 9.15 / PITCH_WIDTH), line, 1)
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

    for det in detections:
        if det.pitch_x is None or det.pitch_y is None:
            continue
        if not (-HALF_LENGTH - 8 <= det.pitch_x <= HALF_LENGTH + 8 and -HALF_WIDTH - 6 <= det.pitch_y <= HALF_WIDTH + 6):
            continue
        px, py = pitch_to_panel(det.pitch_x, det.pitch_y, origin, size)
        if det.role not in {"ball", "referee", "player", "goalkeeper"}:
            continue
        color = role_color(det)
        radius = 3 if det.role in {"ball", "referee"} else 4
        if det.team == "right":
            cv2.circle(frame, (px, py), radius + 1, COLORS["black"], -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)

    legend_y = oy + height + 25
    cv2.circle(frame, (ox + 6, legend_y - 5), 4, COLORS["left"], -1, cv2.LINE_AA)
    put_text(frame, TEAM_LABELS["left"], (ox + 15, legend_y), scale=0.38, color=COLORS["white"])
    cv2.circle(frame, (ox + 79, legend_y - 5), 5, COLORS["black"], -1, cv2.LINE_AA)
    cv2.circle(frame, (ox + 79, legend_y - 5), 4, COLORS["right"], -1, cv2.LINE_AA)
    put_text(frame, TEAM_LABELS["right"], (ox + 88, legend_y), scale=0.38, color=COLORS["white"])
    cv2.circle(frame, (ox + 152, legend_y - 5), 3, COLORS["referee"], -1, cv2.LINE_AA)
    put_text(frame, "裁判", (ox + 160, legend_y), scale=0.38, color=COLORS["white"])


def current_ball(detections: list[Detection]) -> Detection | None:
    balls = [det for det in detections if det.role == "ball" and det.pitch_x is not None and det.pitch_y is not None]
    if not balls:
        return None
    return balls[0]


def ball_zone(ball: Detection | None) -> tuple[str, str, bool]:
    if ball is None or ball.pitch_x is None or ball.pitch_y is None:
        return "unknown", "unknown", False
    if ball.pitch_x < -HALF_LENGTH / 3:
        third = "left third"
    elif ball.pitch_x > HALF_LENGTH / 3:
        third = "right third"
    else:
        third = "middle third"

    if ball.pitch_y < -HALF_WIDTH / 3:
        channel = "bottom wing"
    elif ball.pitch_y > HALF_WIDTH / 3:
        channel = "top wing"
    else:
        channel = "central"

    danger = abs(ball.pitch_x) >= HALF_LENGTH - 16.5 and abs(ball.pitch_y) <= 20.16
    return third, channel, danger


def possession_team(detections: list[Detection], ball: Detection | None) -> str:
    if ball is None or ball.pitch_x is None or ball.pitch_y is None:
        return "unknown"
    best_team = None
    best_dist = 999.0
    for det in detections:
        if det.team not in {"left", "right"} or det.pitch_x is None or det.pitch_y is None:
            continue
        dist = float(np.hypot(det.pitch_x - ball.pitch_x, det.pitch_y - ball.pitch_y))
        if dist < best_dist:
            best_team = det.team
            best_dist = dist
    if best_team is None or best_dist > 8.0:
        return "loose"
    return str(best_team)


def possession_proxy(detections: list[Detection], ball: Detection | None) -> str:
    team = possession_team(detections, ball)
    if team in {"unknown", "loose"}:
        return team
    best_dist = 999.0
    for det in detections:
        if det.team != team or det.pitch_x is None or det.pitch_y is None or ball is None:
            continue
        dist = float(np.hypot(det.pitch_x - ball.pitch_x, det.pitch_y - ball.pitch_y))
        best_dist = min(best_dist, dist)
    return f"{team} ({best_dist:.1f}m)"


def team_shapes(detections: list[Detection]) -> dict[str, dict[str, float]]:
    shapes: dict[str, dict[str, float]] = {}
    for team in ("left", "right"):
        points = [
            (float(det.pitch_x), float(det.pitch_y))
            for det in detections
            if det.team == team
            and det.role in {"player", "goalkeeper"}
            and det.pitch_x is not None
            and det.pitch_y is not None
            and -HALF_LENGTH <= det.pitch_x <= HALF_LENGTH
            and -HALF_WIDTH <= det.pitch_y <= HALF_WIDTH
        ]
        if not points:
            continue
        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        shapes[team] = {
            "count": float(len(points)),
            "cx": float(xs.mean()),
            "cy": float(ys.mean()),
            "depth": float(xs.max() - xs.min()) if len(points) > 1 else 0.0,
            "width": float(ys.max() - ys.min()) if len(points) > 1 else 0.0,
        }
    return shapes


def player_points(detections: list[Detection], team: str | None = None) -> list[tuple[float, float]]:
    points = []
    for det in detections:
        if team is not None and det.team != team:
            continue
        if det.team not in {"left", "right"} or det.role not in {"player", "goalkeeper"}:
            continue
        if det.pitch_x is None or det.pitch_y is None:
            continue
        if not (-HALF_LENGTH <= det.pitch_x <= HALF_LENGTH and -HALF_WIDTH <= det.pitch_y <= HALF_WIDTH):
            continue
        points.append((float(det.pitch_x), float(det.pitch_y)))
    return points


def opponent(team: str) -> str:
    return "right" if team == "left" else "left"


def count_near(points: list[tuple[float, float]], center: tuple[float, float], radius: float) -> int:
    return sum(1 for x, y in points if float(np.hypot(x - center[0], y - center[1])) <= radius)


def danger_attack(detections: list[Detection], ball: Detection | None, possession: str) -> dict[str, object]:
    if ball is None or ball.pitch_x is None or ball.pitch_y is None:
        return {"active": False, "label": "no ball"}
    bx, by = float(ball.pitch_x), float(ball.pitch_y)
    in_box = abs(bx) >= HALF_LENGTH - 16.5 and abs(by) <= 20.16
    final_third = abs(bx) >= HALF_LENGTH / 3
    central_lane = abs(by) <= 18.0

    attackers = player_points(detections, possession) if possession in {"left", "right"} else []
    defenders = player_points(detections, opponent(possession)) if possession in {"left", "right"} else []
    near_attackers = count_near(attackers, (bx, by), 18.0)
    near_defenders = count_near(defenders, (bx, by), 14.0)
    overload = near_attackers - near_defenders

    if in_box and near_attackers >= 1:
        label = f"box attack A{near_attackers}/D{near_defenders}"
        return {"active": True, "label": label}
    if final_third and central_lane and overload >= 0:
        label = f"central threat A{near_attackers}/D{near_defenders}"
        return {"active": True, "label": label}
    if final_third:
        return {"active": False, "label": "final third"}
    return {"active": False, "label": "normal"}


def pressure_state(detections: list[Detection], ball: Detection | None, possession: str) -> dict[str, object]:
    if ball is None or ball.pitch_x is None or ball.pitch_y is None or possession not in {"left", "right"}:
        return {"label": "unknown", "near5": 0, "near8": 0, "near12": 0}
    center = (float(ball.pitch_x), float(ball.pitch_y))
    defenders = player_points(detections, opponent(possession))
    near5 = count_near(defenders, center, 5.0)
    near8 = count_near(defenders, center, 8.0)
    near12 = count_near(defenders, center, 12.0)
    if near5 >= 1 or near8 >= 2:
        label = f"high {near5}/{near8}/{near12}"
    elif near8 >= 1 or near12 >= 2:
        label = f"medium {near5}/{near8}/{near12}"
    else:
        label = f"low {near5}/{near8}/{near12}"
    return {"label": label, "near5": near5, "near8": near8, "near12": near12}


def overload_state(detections: list[Detection], ball: Detection | None) -> dict[str, object]:
    if ball is None or ball.pitch_x is None or ball.pitch_y is None:
        return {"label": "unknown"}
    center = (float(ball.pitch_x), float(ball.pitch_y))
    left_count = count_near(player_points(detections, "left"), center, 18.0)
    right_count = count_near(player_points(detections, "right"), center, 18.0)
    diff = left_count - right_count
    if diff >= 2:
        label = f"left +{diff} near ball"
    elif diff <= -2:
        label = f"right +{abs(diff)} near ball"
    else:
        label = f"balanced {left_count}:{right_count}"
    return {"label": label, "left": left_count, "right": right_count}


def team_compactness(detections: list[Detection]) -> dict[str, dict[str, float | str]]:
    result: dict[str, dict[str, float | str]] = {}
    for team in ("left", "right"):
        points = player_points(detections, team)
        if len(points) < 2:
            result[team] = {"distance": 0.0, "label": "unknown"}
            continue
        dists = []
        for i, (x1, y1) in enumerate(points):
            for x2, y2 in points[i + 1:]:
                dists.append(float(np.hypot(x1 - x2, y1 - y2)))
        avg_dist = float(np.mean(dists)) if dists else 0.0
        if avg_dist < 14:
            label = "compact"
        elif avg_dist < 24:
            label = "balanced"
        else:
            label = "stretched"
        result[team] = {"distance": avg_dist, "label": label}
    return result


def counter_percent(counter: Counter, key: str) -> int:
    total = sum(counter.values())
    if total <= 0:
        return 0
    return int(round(counter[key] * 100 / total))


def zh_team(team: str | None) -> str:
    return {
        "left": TEAM_LABELS["left"],
        "right": TEAM_LABELS["right"],
        "loose": "争抢中",
        "unknown": "未知",
    }.get(str(team), str(team))


def zh_zone(label: str) -> str:
    return {
        "left third": "左侧三区",
        "middle third": "中场三区",
        "right third": "右侧三区",
        "top wing": "上边路",
        "central": "中路",
        "bottom wing": "下边路",
        "unknown": "未知",
    }.get(label, label)


def zh_compactness(label: object) -> str:
    return {"compact": "紧凑", "balanced": "均衡", "stretched": "拉开", "unknown": "未知"}.get(str(label), str(label))


def zh_pressure(label: str) -> str:
    if label.startswith("high"):
        return "高压"
    if label.startswith("medium"):
        return "中压"
    if label.startswith("low"):
        return "低压"
    return "未知"


def zh_danger(label: str) -> str:
    if label.startswith("box attack"):
        return "禁区进攻"
    if label.startswith("central threat"):
        return "中路威胁"
    if label == "final third":
        return "进入前场"
    if label == "normal":
        return "普通推进"
    return "无球"


def zh_possession(text: str) -> str:
    for team in ("left", "right", "loose", "unknown"):
        if text.startswith(team):
            return text.replace(team, zh_team(team), 1)
    return text


def draw_info_panel(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    detections: list[Detection],
    ball: Detection | None,
    state: TacticalState,
) -> None:
    counts = Counter()
    for det in detections:
        if det.role == "ball":
            counts["ball"] += 1
        elif det.role == "referee":
            counts["referee"] += 1
        elif det.team in {"left", "right"}:
            counts[det.team] += 1

    third, channel, danger = ball_zone(ball)
    possession = possession_proxy(detections, ball)
    shapes = state.latest_shapes
    left_shape = shapes.get("left", {})
    right_shape = shapes.get("right", {})
    compact_l = state.latest_compactness.get("left", {})
    compact_r = state.latest_compactness.get("right", {})
    x1, y1, x2, y2 = 20, frame.shape[0] - 246, 650, frame.shape[0] - 18
    blend_rect(frame, x1, y1, x2, y2, COLORS["panel"], 0.70)
    conclusion = f"{zh_team(possession_team(detections, ball))}控球，{zh_danger(state.latest_danger)}，防守压力{zh_pressure(state.latest_pressure)}"
    put_text(frame, "本段战术结论", (x1 + 14, y1 + 30), scale=0.64, thickness=2)
    put_text(frame, conclusion, (x1 + 14, y1 + 62), scale=0.54, thickness=1)
    put_text(frame, f"时间 {frame_idx / fps:05.2f}s  画面帧 {frame_idx}", (x1 + 14, y1 + 92), scale=0.44)
    put_text(frame, f"控球占比：{TEAM_LABELS['left']} {counter_percent(state.possession, 'left')}%  {TEAM_LABELS['right']} {counter_percent(state.possession, 'right')}%  争抢 {counter_percent(state.possession, 'loose')}%", (x1 + 14, y1 + 120), scale=0.44)
    put_text(frame, f"区域：{zh_zone(third)} / {zh_zone(channel)}    危险进攻帧：{state.danger_attacks}", (x1 + 14, y1 + 148), scale=0.44)
    put_text(frame, f"转换：{state.transitions} 次    球速：{state.ball_speed:04.1f} m/s    当前球权：{zh_possession(possession)}", (x1 + 14, y1 + 176), scale=0.44)
    put_text(frame, f"站位紧凑度：{TEAM_LABELS['left']} {zh_compactness(compact_l.get('label', 'unknown'))} / {TEAM_LABELS['right']} {zh_compactness(compact_r.get('label', 'unknown'))}", (x1 + 14, y1 + 204), scale=0.44)
    put_text(frame, f"可见人数：{TEAM_LABELS['left']} {counts['left']:02d}  {TEAM_LABELS['right']} {counts['right']:02d}  裁判 {counts['referee']}", (x1 + 392, y1 + 92), scale=0.42)
    if danger:
        cv2.rectangle(frame, (x2 - 112, y1 + 14), (x2 - 14, y1 + 44), COLORS["danger"], -1)
        put_text(frame, "危险", (x2 - 94, y1 + 37), scale=0.54, thickness=2)


def draw_frame(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    detections: list[Detection],
    ball_trail: deque[tuple[float, float]],
    state: TacticalState,
    fallback_detections: list[Detection] | None = None,
) -> np.ndarray:
    output = frame.copy()
    fallback_detections = fallback_detections or []
    ball = current_ball(detections)
    if ball is not None and ball.pitch_x is not None and ball.pitch_y is not None:
        ball_trail.append((ball.pitch_x, ball.pitch_y))
    state.update(detections, ball, fps)

    for det in detections + fallback_detections:
        if not det.bbox:
            continue
        x = int(det.bbox.get("x", 0))
        y = int(det.bbox.get("y", 0))
        w = int(det.bbox.get("w", 0))
        h = int(det.bbox.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        color = role_color(det)
        thickness = 3 if det.role == "ball" else 2
        if det.team == "right":
            cv2.rectangle(output, (x - 1, y - 1), (x + w + 1, y + h + 1), COLORS["black"], thickness + 2)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        if det.role == "ball":
            label = "球"
        elif det.role == "fallback_ball":
            label = "球*"
        elif det.role == "fallback_person":
            label = "人*"
        elif det.role == "referee":
            label = f"裁判 {det.track_id}"
        elif det.team in {"left", "right"}:
            label = f"{zh_team(det.team)} {det.track_id}"
        else:
            label = f"待确认 {det.track_id}"
        put_text(output, label, (x, max(16, y - 6)), scale=0.46, color=color, thickness=1)

    pitch_w = max(280, output.shape[1] // 4)
    pitch_h = int(pitch_w * PITCH_WIDTH / PITCH_LENGTH)
    origin = (output.shape[1] - pitch_w - 22, 46)
    draw_mini_pitch(output, origin, (pitch_w, pitch_h), detections, ball_trail, state)
    draw_info_panel(output, frame_idx, fps, detections, ball, state)
    return output


def main() -> int:
    args = parse_args()
    TEAM_LABELS["left"] = args.team0_label
    TEAM_LABELS["right"] = args.team1_label
    COLORS["left"] = DISPLAY_COLORS[args.team0_display_color]
    COLORS["right"] = DISPLAY_COLORS[args.team1_display_color]
    input_video = (PROJECT_ROOT / args.input_video).resolve()
    json_path = (PROJECT_ROOT / args.json_path).resolve()
    output_video = (PROJECT_ROOT / args.output_video).resolve()

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
    fallback_model = load_fallback_model(args)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")

    snapshot_path = output_video.with_name(f"{output_video.stem}_frame{args.snapshot_frame:03d}.jpg")
    ball_trail: deque[tuple[float, float]] = deque(maxlen=args.trail_frames)
    tactical_state = TacticalState()
    frame_idx = 1
    wrote_snapshot = False
    while frame_idx <= frame_limit:
        ok, frame = cap.read()
        if not ok:
            break
        detections = by_frame.get(frame_idx, [])
        player_like_count = sum(1 for det in detections if det.role not in {"ball"})
        fallback = []
        if args.yolo_fallback and player_like_count < args.fallback_min_boxes:
            fallback = fallback_detections(
                fallback_model,
                args.fallback_backend,
                frame,
                detections,
                args.fallback_conf,
            )
        visualized = draw_frame(frame, frame_idx, fps, detections, ball_trail, tactical_state, fallback)
        writer.write(visualized)
        if frame_idx == args.snapshot_frame:
            cv2.imwrite(str(snapshot_path), visualized)
            wrote_snapshot = True
        frame_idx += 1

    cap.release()
    writer.release()
    written = frame_idx - 1
    if not wrote_snapshot and written > 0:
        print(f"Snapshot frame {args.snapshot_frame} was outside processed range.")
    print(f"Wrote {written} frames at {fps:.2f} fps to {output_video}")
    if wrote_snapshot:
        print(f"Wrote snapshot to {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
