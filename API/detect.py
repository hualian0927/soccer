#!/usr/bin/env python3
"""Stage 1: run RF-DETR with Global SAHI and save frame detections."""

from __future__ import annotations

import argparse
from pathlib import Path

from football_inference import detect_video_sahi


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PACKAGE_ROOT / "model" / "rfdetr_ball_best_ema.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Detection output directory.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="RF-DETR checkpoint.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument("--tile-size", type=int, default=672)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    args = parser.parse_args()

    result = detect_video_sahi(
        args.input,
        args.output_dir,
        device=args.device,
        confidence=args.confidence,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        overlap=args.overlap,
        nms_iou=args.nms_iou,
        rfdetr_weights=args.weights,
    )
    print(f"Detection result: {result}")


if __name__ == "__main__":
    main()
