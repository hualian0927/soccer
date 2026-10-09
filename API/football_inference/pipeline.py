from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETR
from sahi.slicing import slice_image

from .tcn_model import BidirectionalTrajectoryTCN


def emit_progress(stage: str, progress: float, processed_frames: int, total_frames: int) -> None:
    print(
        "PROGRESS "
        + json.dumps(
            {
                "stage": stage,
                "progress": round(progress, 2),
                "processed_frames": processed_frames,
                "total_frames": total_frames,
            }
        ),
        flush=True,
    )


def video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    info = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    if min(info["width"], info["height"], info["frames"]) <= 0:
        raise RuntimeError("视频元数据无效。")
    return info


def open_writer(path: Path, info: dict):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["width"], info["height"])
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建结果视频：{path}")
    return writer


def nms(boxes: list[np.ndarray], scores: list[float], threshold: float) -> list[int]:
    if not boxes:
        return []
    data = np.asarray(boxes, dtype=np.float32)
    areas = np.maximum(0, data[:, 2] - data[:, 0]) * np.maximum(0, data[:, 3] - data[:, 1])
    order = np.argsort(np.asarray(scores))[::-1]
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        xx1 = np.maximum(data[current, 0], data[remaining, 0])
        yy1 = np.maximum(data[current, 1], data[remaining, 1])
        xx2 = np.minimum(data[current, 2], data[remaining, 2])
        yy2 = np.minimum(data[current, 3], data[remaining, 3])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[current] + areas[remaining] - intersection
        order = remaining[intersection / np.maximum(union, 1e-6) <= threshold]
    return keep


def predict_tiles(
    model: Any,
    images: list[np.ndarray],
    offsets: list[tuple[int, int]],
    confidence: float,
) -> list[tuple[np.ndarray, float]]:
    predictions = model.predict(images, threshold=confidence, include_source_image=False)
    detections = []
    for prediction, (offset_x, offset_y) in zip(predictions, offsets):
        for box, score, class_id in zip(prediction.xyxy, prediction.confidence, prediction.class_id):
            if int(class_id) != 0:
                continue
            shifted = np.asarray(box, dtype=np.float32).copy()
            shifted[[0, 2]] += offset_x
            shifted[[1, 3]] += offset_y
            detections.append((shifted, float(score)))
    return detections


def detect_video(
    input_path: Path,
    output_dir: Path,
    weights: Path,
    device: str,
    confidence: float,
    resolution: int,
    batch_size: int = 4,
    overlap: float = 0.2,
    nms_iou: float = 0.5,
):
    if not weights.is_file():
        raise FileNotFoundError(f"RF-DETR 权重不存在：{weights}")
    info = video_info(input_path)
    model = RFDETR.from_checkpoint(str(weights), device=device, trust_checkpoint=True)
    if hasattr(model, "model_config"):
        model.model_config.positional_encoding_size = 48
    if device.startswith("cuda"):
        model.inference(compile=False, dtype="float16", inplace=True)

    capture = cv2.VideoCapture(str(input_path))
    silent_path = output_dir / "detection_silent.mp4"
    writer = open_writer(silent_path, info)
    records = []
    tile_count = 0
    detection_count = 0
    inference_seconds = 0.0
    for frame_index in range(info["frames"]):
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sliced = slice_image(
            Image.fromarray(rgb),
            slice_height=resolution,
            slice_width=resolution,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            auto_slice_resolution=False,
            verbose=False,
        )
        candidates: list[tuple[np.ndarray, float]] = []
        for start in range(0, len(sliced.images), batch_size):
            images = [np.array(image, copy=True) for image in sliced.images[start:start + batch_size]]
            offsets = [tuple(offset) for offset in sliced.starting_pixels[start:start + batch_size]]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            candidates.extend(predict_tiles(model, images, offsets, confidence))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started
        keep = nms([item[0] for item in candidates], [item[1] for item in candidates], nms_iou)
        candidates = [candidates[index] for index in keep]
        selected = max(candidates, key=lambda item: item[1], default=None)
        serialized_detections = [
            {"box": box.tolist(), "confidence": score} for box, score in candidates
        ]
        for box, score in candidates:
            x1, y1, x2, y2 = np.rint(box).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)
            cv2.putText(frame, f"ball {score:.2f}", (x1, max(24, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (34, 197, 94), 2, cv2.LINE_AA)
        if selected:
            box, score = selected
            center = [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)]
            box_values = box.tolist()
        else:
            box_values = center = None
            score = None
        records.append(
            {
                "frame_index": frame_index,
                "box": box_values,
                "center": center,
                "confidence": score,
                "observed": selected is not None,
                "interpolated": False,
                "candidate_count": len(candidates),
                "detections": serialized_detections,
            }
        )
        cv2.putText(frame, f"frame {frame_index} | RF-DETR + SAHI", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        tile_count += len(sliced.images)
        detection_count += len(candidates)
        if frame_index % 20 == 0:
            emit_progress("sahi_detect", 100 * (frame_index + 1) / info["frames"], frame_index + 1, info["frames"])
    capture.release()
    writer.release()
    info["frames"] = len(records)
    finalize_video(silent_path, input_path, output_dir / "detection.mp4")
    settings = {
        "algorithm": "RF-DETR + Global SAHI",
        "confidence": confidence,
        "tile_size": resolution,
        "overlap": overlap,
        "nms_iou": nms_iou,
        "batch_size": batch_size,
        "target_class_id": 0,
        "total_sahi_tiles": tile_count,
        "detections_after_nms": detection_count,
        "inference_seconds": inference_seconds,
    }
    write_detections(output_dir, records, info, settings)
    emit_progress("sahi_detect_completed", 100, info["frames"], info["frames"])
    return records, info, settings


def write_detections(output_dir: Path, records: list[dict], info: dict, settings: dict) -> None:
    payload = {"schema_version": 1, "video": info, "settings": settings, "records": records}
    (output_dir / "detections.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    with (output_dir / "detections.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_ms", "x1", "y1", "x2", "y2", "confidence", "selected"])
        for record in records:
            detections = record["detections"]
            if not detections:
                writer.writerow([record["frame_index"], round(record["frame_index"] / info["fps"] * 1000),
                                 "", "", "", "", "", False])
            for index, detection in enumerate(detections):
                writer.writerow([record["frame_index"], round(record["frame_index"] / info["fps"] * 1000),
                                 *detection["box"], detection["confidence"], index == 0])


def load_detections(path: Path) -> tuple[list[dict], dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Detection result does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError(f"Unsupported detection result format: {path}")
    return payload["records"], payload["video"], payload.get("settings", {})


def records_to_features(records: list[dict], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((len(records), 16), dtype=np.float32)
    valid = np.zeros(len(records), dtype=bool)
    for index, record in enumerate(records):
        center = record["center"]
        box = record["box"]
        if center is not None:
            features[index, 0:2] = [center[0] / width, center[1] / height]
            features[index, 6] = float(record["confidence"] or 0)
            features[index, 7] = 1.0
            features[index, 15] = 1.0
            valid[index] = True
        if box is not None:
            features[index, 4:6] = [(box[2] - box[0]) / width, (box[3] - box[1]) / height]
        features[index, 9] = min(1.0, record["candidate_count"] / 5.0)
        features[index, 13] = 0.0
    for index in range(1, len(records)):
        if valid[index] and valid[index - 1]:
            features[index, 2:4] = features[index, 0:2] - features[index - 1, 0:2]
    return features, valid


def complete_trajectory(records: list[dict], info: dict, checkpoint_path: Path, device_name: str):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TCN 权重不存在：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    radius = int(config["window_radius"])
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = BidirectionalTrajectoryTCN(
        config["feature_dim"], config["hidden_dim"], config["dilations"], config["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    features, valid = records_to_features(records, info["width"], info["height"])
    scale = np.asarray([info["width"], info["height"]], dtype=np.float32)
    results = []
    batch_items = []

    def flush():
        if not batch_items:
            return
        batch = torch.from_numpy(np.stack([item[1] for item in batch_items])).to(device)
        trajectory = batch.clone()
        trajectory[:, radius, 0:11] = 0
        trajectory[:, radius, 15] = 0
        with torch.no_grad():
            predicted = model(batch, trajectory)
        centers = predicted["trajectory_center"].cpu().numpy() * scale
        qualities = predicted["quality"].cpu().numpy()[:, 0]
        for batch_index, (record_index, window, context_valid) in enumerate(batch_items):
            record = records[record_index]
            tcn_center = centers[batch_index]
            in_bounds = bool(
                context_valid
                and 0 <= tcn_center[0] < info["width"]
                and 0 <= tcn_center[1] < info["height"]
            )
            if record["center"] is not None:
                final_center = record["center"]
                output_status = "detected"
            elif in_bounds:
                final_center = tcn_center.tolist()
                output_status = "completed_missing"
            else:
                final_center = None
                output_status = "invalid"
            results.append(
                {
                    **record,
                    "timestamp_ms": round(record_index / info["fps"] * 1000),
                    "tcn_center": tcn_center.tolist() if in_bounds else None,
                    "final_center": final_center,
                    "output_status": output_status,
                    "observation_quality": float(qualities[batch_index]),
                }
            )
        batch_items.clear()

    for index in range(len(records)):
        left, right = index - radius, index + radius + 1
        window = np.zeros((2 * radius + 1, features.shape[1]), dtype=np.float32)
        source_left, source_right = max(0, left), min(len(records), right)
        destination_left = source_left - left
        window[destination_left:destination_left + source_right - source_left] = features[source_left:source_right]
        batch_items.append((index, window, bool(valid[source_left:source_right].any())))
        if len(batch_items) >= 256:
            flush()
            emit_progress("tcn_complete", 60 * (index + 1) / len(records), index + 1, len(records))
    flush()
    return results


def draw_completion(input_path: Path, output_path: Path, results: list[dict], info: dict):
    capture = cv2.VideoCapture(str(input_path))
    silent_path = output_path.with_name("tcn_completed_silent.mp4")
    writer = open_writer(silent_path, info)
    tail = deque(maxlen=18)
    for index, result in enumerate(results):
        ok, frame = capture.read()
        if not ok:
            break
        point = result["final_center"]
        if point is not None:
            center = tuple(np.rint(point).astype(int))
            if tail and np.linalg.norm(np.asarray(center) - np.asarray(tail[-1])) > 300:
                tail.clear()
            tail.append(center)
            points = list(tail)
            for tail_index in range(1, len(points)):
                cv2.line(frame, points[tail_index - 1], points[tail_index], (46, 204, 113), 2, cv2.LINE_AA)
            color = (34, 197, 94) if result["output_status"] == "detected" else (245, 158, 11)
            cv2.circle(frame, center, 9, color, -1, cv2.LINE_AA)
            cv2.circle(frame, center, 12, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            tail.clear()
        writer.write(frame)
        if index % 30 == 0:
            emit_progress("render", 60 + 35 * (index + 1) / len(results), index + 1, len(results))
    capture.release()
    writer.release()
    finalize_video(silent_path, input_path, output_path)


def finalize_video(silent_path: Path, source_path: Path, output_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        silent_path.replace(output_path)
        return
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(silent_path), "-i", str(source_path),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", "-shortest", str(output_path),
    ]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if completed.returncode == 0:
        silent_path.unlink(missing_ok=True)
        return
    output_path.unlink(missing_ok=True)
    silent_path.replace(output_path)


def write_results(output_dir: Path, results: list[dict], info: dict, settings_snapshot: dict):
    jsonl_path = output_dir / "trajectory.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")

    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "frame_index", "timestamp_ms", "detection_x", "detection_y", "detection_confidence",
            "detection_x1", "detection_y1", "detection_x2", "detection_y2",
            "tcn_x", "tcn_y", "final_x", "final_y", "output_status", "algorithm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            detection = row["center"] or [None, None]
            box = row["box"] or [None, None, None, None]
            tcn = row["tcn_center"] or [None, None]
            final = row["final_center"] or [None, None]
            writer.writerow(
                {
                    "frame_index": row["frame_index"], "timestamp_ms": row["timestamp_ms"],
                    "detection_x": detection[0], "detection_y": detection[1],
                    "detection_confidence": row["confidence"],
                    "detection_x1": box[0], "detection_y1": box[1],
                    "detection_x2": box[2], "detection_y2": box[3],
                    "tcn_x": tcn[0], "tcn_y": tcn[1],
                    "final_x": final[0], "final_y": final[1], "output_status": row["output_status"],
                    "algorithm": "RF-DETR + TCN",
                }
            )
    counts = {status: sum(row["output_status"] == status for row in results) for status in ["detected", "completed_missing", "invalid"]}
    summary = {"algorithm": "RF-DETR + TCN", "video": info, "frames": len(results), "status_counts": counts,
               "settings": settings_snapshot}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_detection(
    input_path: Path,
    output_dir: Path,
    rfdetr_weights: Path,
    device: str,
    confidence: float = 0.18,
    resolution: int = 672,
    batch_size: int = 4,
    overlap: float = 0.2,
    nms_iou: float = 0.5,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    detect_video(input_path, output_dir, rfdetr_weights, device, confidence, resolution,
                 batch_size, overlap, nms_iou)
    return output_dir / "detections.json"


def run_completion(
    input_path: Path,
    detections_path: Path,
    output_dir: Path,
    tcn_weights: Path,
    device: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records, info, detection_settings = load_detections(detections_path)
    actual_info = video_info(input_path)
    for field in ("width", "height", "frames"):
        if actual_info[field] != info[field]:
            raise ValueError(
                f"Input video does not match detections.json: {field} is "
                f"{actual_info[field]}, expected {info[field]}."
            )
    if abs(actual_info["fps"] - info["fps"]) > 0.01:
        raise ValueError(
            f"Input video does not match detections.json: fps is {actual_info['fps']}, expected {info['fps']}."
        )
    emit_progress("tcn_complete", 0, 0, info["frames"])
    results = complete_trajectory(records, info, tcn_weights, device)
    emit_progress("render", 60, 0, info["frames"])
    draw_completion(input_path, output_dir / "tcn_completed.mp4", results, info)
    write_results(
        output_dir,
        results,
        info,
        {"detector": detection_settings, "final_rule": "detection_first_tcn_on_missing"},
    )
    emit_progress("completed", 100, info["frames"], info["frames"])


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    rfdetr_weights: Path,
    tcn_weights: Path,
    device: str,
    confidence: float,
    resolution: int,
    batch_size: int = 4,
    overlap: float = 0.2,
    nms_iou: float = 0.5,
) -> None:
    detections_path = run_detection(input_path, output_dir, rfdetr_weights, device, confidence,
                                    resolution, batch_size, overlap, nms_iou)
    run_completion(input_path, detections_path, output_dir, tcn_weights, device)
