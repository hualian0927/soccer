"""SoccerNetGSR adapter for the shared analysis data model."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import AnalysisContext, FrameState, Observation


def parse_frame(image_id: Any) -> int:
    text = str(image_id or "0")
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return 0
    return int(digits[-6:])


def normalize_team(value: Any) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in {"left", "right"} else None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def load_gsr_context(
    json_path: Path,
    fps: float = 25.0,
    video_path: Path | None = None,
) -> AnalysisContext:
    with json_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    by_frame: dict[int, list[Observation]] = defaultdict(list)
    invalid_predictions = 0
    for prediction in raw.get("predictions", []):
        frame = parse_frame(prediction.get("image_id"))
        if frame <= 0:
            invalid_predictions += 1
            continue
        attributes = prediction.get("attributes") or {}
        pitch = prediction.get("bbox_pitch") or {}
        bbox_raw = prediction.get("bbox_image") or None
        bbox = None
        if bbox_raw:
            bbox = {
                key: value
                for key in ("x", "y", "w", "h")
                if (value := _number(bbox_raw.get(key))) is not None
            }
        by_frame[frame].append(
            Observation(
                frame=frame,
                time_sec=(frame - 1) / max(fps, 1e-6),
                track_id=int(prediction.get("track_id", -1)),
                role=str(attributes.get("role", "unknown")).strip().lower(),
                team=normalize_team(attributes.get("team")),
                pitch_x=_number(pitch.get("x_bottom_middle")),
                pitch_y=_number(pitch.get("y_bottom_middle")),
                bbox=bbox,
                confidence=_number(prediction.get("confidence") or prediction.get("score")),
            )
        )

    frames = [
        FrameState(frame=frame, time_sec=(frame - 1) / max(fps, 1e-6), observations=observations)
        for frame, observations in sorted(by_frame.items())
    ]
    metadata = {
        "adapter": "SoccerNetGSR",
        "prediction_count": sum(len(frame.observations) for frame in frames),
        "observed_frame_count": len(frames),
        "first_frame": frames[0].frame if frames else None,
        "last_frame": frames[-1].frame if frames else None,
        "invalid_prediction_count": invalid_predictions,
    }
    return AnalysisContext(
        source_json=json_path,
        source_video=video_path,
        fps=fps,
        frames=frames,
        metadata=metadata,
    )


def infer_video_fps(video_path: Path | None, fallback: float = 25.0) -> float:
    if video_path is None:
        return fallback
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
        capture.release()
        return fps if fps > 0 else fallback
    except Exception:
        return fallback
