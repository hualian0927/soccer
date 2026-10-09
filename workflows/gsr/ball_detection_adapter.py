"""Bridge the packaged RF-DETR/TCN football trajectory to SoccerNetGSR JSON."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


def make_segment_video(image_dir: Path, output_path: Path, fps: float) -> int:
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise ValueError(f"No extracted frames in {image_dir}")
    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError(f"Cannot read {images[0]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {output_path}")
    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(f"Invalid extracted frame: {image_path}")
            writer.write(frame)
    finally:
        writer.release()
    return len(images)


def project_ball_center(matrix: np.ndarray, x: float, y: float, center: tuple[float, float]) -> dict:
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return {}
    point = matrix @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(point[2]) < 1e-8:
        return {}
    pitch_x = float(point[0] / point[2] - center[0])
    pitch_y = float(point[1] / point[2] - center[1])
    if not (math.isfinite(pitch_x) and math.isfinite(pitch_y)):
        return {}
    if abs(pitch_x) > 80 or abs(pitch_y) > 55:
        return {}
    return {
        "x_bottom_left": pitch_x, "y_bottom_left": pitch_y,
        "x_bottom_right": pitch_x, "y_bottom_right": pitch_y,
        "x_bottom_middle": pitch_x, "y_bottom_middle": pitch_y,
    }


def merge_ball_trajectory(
    gsr_json: Path,
    trajectory_jsonl: Path,
    homography_dir: Path,
    template_path: Path,
    video_id: str,
    expected_frames: int,
) -> dict[str, int]:
    """Replace only football observations, preserving all player/referee records."""
    with gsr_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("SoccerNetGSR JSON has no predictions array")
    template = cv2.imread(str(template_path))
    if template is None:
        raise FileNotFoundError(template_path)
    center = (template.shape[1] / 2.0, template.shape[0] / 2.0)
    with trajectory_jsonl.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != expected_frames:
        raise ValueError(f"Trajectory has {len(rows)} frames, expected {expected_frames}")
    if any(int(row.get("frame_index", -1)) != index for index, row in enumerate(rows)):
        raise ValueError("Trajectory frame indexes must be continuous and zero-based")

    retained = [item for item in predictions if (item.get("attributes") or {}).get("role") != "ball"]
    removed = len(predictions) - len(retained)
    box_sizes = [
        (float(row["box"][2]) - float(row["box"][0]), float(row["box"][3]) - float(row["box"][1]))
        for row in rows if row.get("box") is not None
    ]
    default_width = min(18.0, max(3.0, float(np.median([w for w, _ in box_sizes])))) if box_sizes else 8.0
    default_height = min(18.0, max(3.0, float(np.median([h for _, h in box_sizes])))) if box_sizes else 8.0
    counts = {"detected": 0, "completed_missing": 0, "invalid": 0, "removed_legacy_balls": removed}
    next_id = max((int(item.get("id", 0)) for item in predictions if str(item.get("id", "")).isdigit()), default=0) + 1
    for index, row in enumerate(rows):
        status = str(row.get("output_status", "invalid"))
        if status not in counts:
            raise ValueError(f"Unknown trajectory status: {status}")
        point = row.get("final_center")
        if status == "invalid" or point is None:
            counts["invalid"] += 1
            continue
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            counts["invalid"] += 1
            continue
        frame_number = index + 1
        matrix_path = homography_dir / f"{frame_number:06d}.npy"
        pitch = project_ball_center(np.load(matrix_path), x, y, center) if matrix_path.exists() else {}
        box = row.get("box")
        if box is not None and status == "detected":
            left, top, right, bottom = map(float, box)
            bbox = {"x": left, "y": top, "w": max(1.0, right - left), "h": max(1.0, bottom - top)}
        else:
            bbox = {"x": x - default_width / 2, "y": y - default_height / 2,
                    "w": default_width, "h": default_height}
        retained.append({
            "bbox_pitch": pitch,
            "bbox_image": bbox,
            "category_id": 1.0,
            "image_id": f"3{video_id}{frame_number:06d}",
            "video_id": video_id,
            "track_id": 900000,
            "supercategory": "object",
            "attributes": {"role": "ball", "jersey": None, "team": None, "color": None,
                           "detection_status": status, "detector": "RF-DETR + TCN"},
            "confidence": float(row.get("confidence") or 0.0) if status == "detected" else 0.0,
            "id": str(next_id),
        })
        next_id += 1
        counts[status] += 1
    if counts["detected"] + counts["completed_missing"] == 0:
        raise ValueError("RF-DETR/TCN produced no valid ball positions; original JSON was not changed")
    payload["predictions"] = retained
    temporary = gsr_json.with_suffix(".json.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(temporary, gsr_json)
    finally:
        temporary.unlink(missing_ok=True)
    return counts
