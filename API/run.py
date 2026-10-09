#!/usr/bin/env python3
"""Command-line entry point for football detection and trajectory completion."""

from __future__ import annotations

import argparse
from pathlib import Path

from football_inference import infer_video


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RFDETR_WEIGHTS = PACKAGE_ROOT / "model" / "rfdetr_ball_best_ema.pth"
DEFAULT_TCN_WEIGHTS = PACKAGE_ROOT / "model" / "tcn_ball_completion_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-DETR football detection and TCN trajectory completion."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--rfdetr-weights", type=Path, default=DEFAULT_RFDETR_WEIGHTS)
    parser.add_argument("--tcn-weights", type=Path, default=DEFAULT_TCN_WEIGHTS)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. 'auto' uses CUDA when available.",
    )
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument("--resolution", type=int, default=672)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video does not exist: {args.input}")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1.")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")

    infer_video(
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        rfdetr_weights=args.rfdetr_weights,
        tcn_weights=args.tcn_weights,
        device=args.device,
        confidence=args.confidence,
        resolution=args.resolution,
        batch_size=args.batch_size,
        overlap=args.overlap,
        nms_iou=args.nms_iou,
    )


if __name__ == "__main__":
    main()
