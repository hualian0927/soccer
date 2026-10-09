"""Public Python interface with package-relative model defaults."""

from __future__ import annotations

from pathlib import Path

import torch

from .pipeline import run_completion, run_detection, run_pipeline


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PACKAGE_ROOT / "model"


def infer_video(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    confidence: float = 0.18,
    resolution: int = 672,
    batch_size: int = 4,
    overlap: float = 0.2,
    nms_iou: float = 0.5,
    rfdetr_weights: str | Path | None = None,
    tcn_weights: str | Path | None = None,
) -> None:
    """Run RF-DETR detection followed by TCN trajectory completion."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    detector = Path(rfdetr_weights).expanduser().resolve() if rfdetr_weights else MODEL_DIR / "rfdetr_ball_best_ema.pth"
    trajectory = Path(tcn_weights).expanduser().resolve() if tcn_weights else MODEL_DIR / "tcn_ball_completion_best.pt"

    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    _validate_detection_options(confidence, resolution, batch_size, overlap, nms_iou)
    device = _resolve_device(device)

    run_pipeline(source, destination, detector, trajectory, device, confidence, resolution,
                 batch_size, overlap, nms_iou)


def detect_video_sahi(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    confidence: float = 0.18,
    tile_size: int = 672,
    batch_size: int = 4,
    overlap: float = 0.2,
    nms_iou: float = 0.5,
    rfdetr_weights: str | Path | None = None,
) -> Path:
    """Run RF-DETR with Global SAHI and return the detections.json path."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    detector = Path(rfdetr_weights).expanduser().resolve() if rfdetr_weights else MODEL_DIR / "rfdetr_ball_best_ema.pth"
    device = _resolve_device(device)
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    _validate_detection_options(confidence, tile_size, batch_size, overlap, nms_iou)
    return run_detection(source, destination, detector, device, confidence, tile_size,
                         batch_size, overlap, nms_iou)


def complete_video_trajectory(
    input_path: str | Path,
    detections_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    tcn_weights: str | Path | None = None,
) -> None:
    """Complete a previously detected trajectory with the packaged TCN."""
    source = Path(input_path).expanduser().resolve()
    detections = Path(detections_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    trajectory = Path(tcn_weights).expanduser().resolve() if tcn_weights else MODEL_DIR / "tcn_ball_completion_best.pt"
    device = _resolve_device(device)
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    run_completion(source, detections, destination, trajectory, device)


def _resolve_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    return device


def _validate_detection_options(confidence: float, tile_size: int, batch_size: int,
                                overlap: float, nms_iou: float) -> None:
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")
    if tile_size <= 0 or batch_size <= 0:
        raise ValueError("tile_size and batch_size must be positive.")
    if not 0 <= overlap < 1 or not 0 <= nms_iou <= 1:
        raise ValueError("overlap must be in [0, 1), and nms_iou must be in [0, 1].")
