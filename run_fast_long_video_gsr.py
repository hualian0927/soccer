#!/usr/bin/env python3
"""Build a sparse SoccerNetGSR-compatible track file for long broadcast videos.

This path keeps the existing pitch homographies and analysis contract, but uses
Ultralytics ByteTrack on sampled frames instead of the much heavier YOLOX-X
baseline. It is intended for long-form preview and candidate analysis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from jersey_color import ColorEstimate, estimate_jersey_color


PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = PROJECT_ROOT / "template" / "Radar_Dimen.png"


@dataclass
class TrackRecord:
    frame: int
    track_id: int
    role: str
    team: str | None
    color: str | None
    confidence: float
    bbox: tuple[float, float, float, float]
    pitch: tuple[float, float, float, float, float, float] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast long-video SoccerNetGSR-compatible tracking.")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--homography-dir", required=True, help="Directory containing 000001.npy style homographies.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--video-id", default="4001")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--frame-stride", type=int, default=5, help="Analyze every Nth source frame.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.12)
    parser.add_argument("--team0-colors", default="red")
    parser.add_argument("--team1-colors", default="blue")
    parser.add_argument("--referee-colors", default="black")
    parser.add_argument("--goalkeeper-team0-colors", default="black")
    parser.add_argument("--goalkeeper-team1-colors", default="yellowgreen")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit source frames; 0 processes the full video.")
    return parser.parse_args()


def color_set(value: str) -> set[str]:
    aliases = {
        "yellowgreen": {"yellow", "green"},
        "lightblue": {"blue"},
        "darkblue": {"blue", "black"},
    }
    result: set[str] = set()
    for raw in value.split(","):
        item = raw.strip().lower()
        if item:
            result.update(aliases.get(item, {item}))
    return result


def crop_box(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    left = max(0, min(width, int(x1)))
    top = max(0, min(height, int(y1)))
    right = max(0, min(width, int(x2)))
    bottom = max(0, min(height, int(y2)))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    return intersection / max(left_area + right_area - intersection, 1e-6)


def suppress_duplicate_tracks(tracks: np.ndarray, threshold: float = 0.72) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for track in sorted(tracks, key=lambda item: float(item[5]), reverse=True):
        if any(box_iou(track[:4], existing[:4]) >= threshold for existing in kept):
            continue
        kept.append(track)
    return kept


def project_bbox(
    matrix: np.ndarray | None,
    bbox: tuple[float, float, float, float],
    template_center: tuple[float, float],
) -> tuple[float, float, float, float, float, float] | None:
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    x, y, width, height = bbox
    bottom = y + height
    points = np.array(
        [[x, bottom, 1.0], [x + width, bottom, 1.0], [x + width / 2.0, bottom, 1.0]],
        dtype=np.float64,
    ).T
    transformed = matrix @ points
    denominators = transformed[2]
    if np.any(np.abs(denominators) < 1e-8):
        return None
    projected = (transformed[:2] / denominators).T
    projected[:, 0] -= template_center[0]
    projected[:, 1] -= template_center[1]
    values = tuple(float(value) for value in projected.reshape(-1))
    if not np.isfinite(values).all():
        return None
    middle_x, middle_y = values[4], values[5]
    if not (-80.0 <= middle_x <= 80.0 and -55.0 <= middle_y <= 55.0):
        return None
    return values


def load_homography(directory: Path, frame: int) -> np.ndarray | None:
    path = directory / f"{frame:06d}.npy"
    if not path.is_file():
        return None
    try:
        return np.load(path)
    except (OSError, ValueError):
        return None


def scene_signature(frame: np.ndarray) -> np.ndarray:
    reduced = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).reshape(-1)


def team_score(estimate: ColorEstimate, colors: set[str]) -> float:
    return max((float(estimate.scores.get(color, 0.0)) for color in colors), default=0.0)


def classify_person(
    estimate: ColorEstimate,
    pitch: tuple[float, float, float, float, float, float] | None,
    team0: set[str],
    team1: set[str],
    referee: set[str],
    goalkeeper0: set[str],
    goalkeeper1: set[str],
) -> tuple[str, str | None, str]:
    color = estimate.label
    near_goal = bool(pitch and abs(pitch[4]) >= 36.0 and abs(pitch[5]) <= 23.0)
    if color in team0:
        return "player", "left", color
    if color in team1:
        return "player", "right", color
    if color in goalkeeper1 and color not in referee:
        return "goalkeeper", "right", color
    if color in goalkeeper0 and near_goal:
        return "goalkeeper", "left", color
    if color in referee:
        return "referee", None, color

    scores = {
        "left": team_score(estimate, team0),
        "right": team_score(estimate, team1),
        "referee": team_score(estimate, referee),
        "goalkeeper_left": team_score(estimate, goalkeeper0) if near_goal else 0.0,
        "goalkeeper_right": team_score(estimate, goalkeeper1),
    }
    label, score = max(scores.items(), key=lambda item: item[1])
    if score < 0.035:
        return "player", None, color
    if label == "referee":
        return "referee", None, color
    if label == "goalkeeper_left":
        return "goalkeeper", "left", color
    if label == "goalkeeper_right":
        return "goalkeeper", "right", color
    return "player", label, color


def stabilize_tracks(records: list[TrackRecord]) -> None:
    votes: dict[int, Counter[tuple[str, str | None, str | None]]] = defaultdict(Counter)
    for record in records:
        if record.role == "ball":
            continue
        weight = max(1, int(round(record.confidence * 10)))
        votes[record.track_id][(record.role, record.team, record.color)] += weight
    for record in records:
        if record.role == "ball" or record.track_id not in votes:
            continue
        role, team, color = votes[record.track_id].most_common(1)[0][0]
        record.role = role
        record.team = team
        record.color = color


def prediction(record: TrackRecord, video_id: str, item_id: int) -> dict[str, object]:
    x, y, width, height = record.bbox
    if record.pitch is None:
        pitch = {}
    else:
        left_x, left_y, right_x, right_y, middle_x, middle_y = record.pitch
        pitch = {
            "x_bottom_left": left_x,
            "y_bottom_left": left_y,
            "x_bottom_right": right_x,
            "y_bottom_right": right_y,
            "x_bottom_middle": middle_x,
            "y_bottom_middle": middle_y,
        }
    return {
        "bbox_pitch": pitch,
        "bbox_image": {"x": x, "y": y, "w": width, "h": height},
        "category_id": 1.0,
        "image_id": f"3{video_id}{record.frame:06d}",
        "video_id": video_id,
        "track_id": record.track_id,
        "supercategory": "object",
        "attributes": {
            "role": record.role,
            "jersey": None,
            "team": record.team,
            "color": record.color,
        },
        "confidence": record.confidence,
        "id": str(item_id),
    }


def main() -> int:
    args = parse_args()
    input_video = (PROJECT_ROOT / args.input_video).resolve()
    homography_dir = (PROJECT_ROOT / args.homography_dir).resolve()
    output_json = (PROJECT_ROOT / args.output_json).resolve()
    if not input_video.is_file():
        raise FileNotFoundError(input_video)
    if not homography_dir.is_dir():
        raise FileNotFoundError(homography_dir)

    template = cv2.imread(str(TEMPLATE_PATH))
    if template is None:
        raise FileNotFoundError(TEMPLATE_PATH)
    template_center = (template.shape[1] / 2.0, template.shape[0] / 2.0)
    team0 = color_set(args.team0_colors)
    team1 = color_set(args.team1_colors)
    referee = color_set(args.referee_colors)
    goalkeeper0 = color_set(args.goalkeeper_team0_colors)
    goalkeeper1 = color_set(args.goalkeeper_team1_colors)

    model = YOLO(args.model)
    tracker_config = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    tracker_config.track_high_thresh = max(0.14, args.confidence)
    tracker_config.track_low_thresh = min(0.05, tracker_config.track_high_thresh)
    tracker_config.new_track_thresh = tracker_config.track_high_thresh
    tracker_config.track_buffer = 90
    tracker = BYTETracker(tracker_config, frame_rate=25)
    records: list[TrackRecord] = []
    ball_track_id = 900000
    sampled_count = 0
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {input_video}")
    source_frame = 0
    previous_signature: np.ndarray | None = None
    track_epoch = 0
    batch_frames: list[np.ndarray] = []
    batch_numbers: list[int] = []

    def process_batch() -> None:
        nonlocal sampled_count, previous_signature, track_epoch
        if not batch_frames:
            return
        results = model.predict(
            batch_frames,
            classes=[0, 32],
            conf=args.confidence,
            iou=0.55,
            imgsz=args.imgsz,
            verbose=False,
        )
        for frame, frame_number, result in zip(batch_frames, batch_numbers, results):
            sampled_count += 1
            signature = scene_signature(frame)
            if previous_signature is not None:
                similarity = cv2.compareHist(previous_signature, signature, cv2.HISTCMP_CORREL)
                if similarity < 0.35:
                    tracker.reset()
                    track_epoch += 1
            previous_signature = signature
            matrix = load_homography(homography_dir, frame_number)
            boxes = result.boxes
            if boxes is None:
                continue
            if len(boxes) == 0:
                tracker.update(boxes.cpu().numpy(), frame)
                continue
            tracker.update(boxes.cpu().numpy(), frame)
            tracks = np.asarray([track.result for track in tracker.tracked_stracks], dtype=np.float32)
            ball_candidates: list[tuple[float, TrackRecord]] = []
            for track in suppress_duplicate_tracks(tracks):
                x1, y1, x2, y2, raw_track_id, confidence, raw_class_id = track[:7]
                class_id = int(raw_class_id)
                xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
                bbox = (float(x1), float(y1), max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1)))
                pitch = project_bbox(matrix, bbox, template_center)
                if class_id == 32:
                    record = TrackRecord(frame_number, ball_track_id, "ball", None, None, float(confidence), bbox, pitch)
                    ball_candidates.append((float(confidence), record))
                    continue
                estimate = estimate_jersey_color(crop_box(frame, xyxy))
                role, team, color = classify_person(
                    estimate, pitch, team0, team1, referee, goalkeeper0, goalkeeper1,
                )
                records.append(
                    TrackRecord(
                        frame_number,
                        track_epoch * 10000 + int(raw_track_id),
                        role,
                        team,
                        color,
                        float(confidence),
                        bbox,
                        pitch,
                    )
                )
            if ball_candidates:
                records.append(max(ball_candidates, key=lambda item: item[0])[1])
            if sampled_count % 250 == 0:
                print(f"Processed source frame {frame_number}", flush=True)

    stride = max(1, args.frame_stride)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        source_frame += 1
        if args.max_frames and source_frame > args.max_frames:
            break
        if (source_frame - 1) % stride != 0:
            continue
        batch_frames.append(frame)
        batch_numbers.append(source_frame)
        if len(batch_frames) >= max(1, args.batch_size):
            process_batch()
            batch_frames.clear()
            batch_numbers.clear()
    process_batch()
    capture.release()

    stabilize_tracks(records)
    predictions = [prediction(record, str(args.video_id), index) for index, record in enumerate(records, start=1)]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump({"predictions": predictions}, handle, ensure_ascii=False)
    print(
        f"Wrote {len(predictions)} sparse predictions from {sampled_count} sampled frames to {output_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
