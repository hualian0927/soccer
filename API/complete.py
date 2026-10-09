#!/usr/bin/env python3
"""Stage 2: complete a saved RF-DETR trajectory with the packaged TCN."""

from __future__ import annotations

import argparse
from pathlib import Path

from football_inference import complete_video_trajectory


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PACKAGE_ROOT / "model" / "tcn_ball_completion_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="The original input video.")
    parser.add_argument("--detections", type=Path, required=True, help="detections.json from detect.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Trajectory output directory.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="TCN checkpoint.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    complete_video_trajectory(
        args.input,
        args.detections,
        args.output_dir,
        device=args.device,
        tcn_weights=args.weights,
    )


if __name__ == "__main__":
    main()
