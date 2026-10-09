#!/usr/bin/env python3
"""Visualize one SoccerNetGSR video folder.

This is a small local wrapper around workflows/visualization/visualize_prediction_results.py. It expects
the usual project layout:

  data/SoccerNetGS/test/SNGS-196/
    img1/000001.jpg
    SNGS-196.json

Example:
  python -m workflows.visualization.visualize_local_result --video-dir data/SoccerNetGS/test/SNGS-196
"""

from __future__ import annotations

import argparse
from pathlib import Path

from workflows.visualization.visualize_prediction_results import visualize_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a single SoccerNetGSR prediction JSON.")
    parser.add_argument(
        "--video-dir",
        default="data/SoccerNetGS/test/SNGS-196",
        help="Directory containing img1/ and <video-name>.json.",
    )
    parser.add_argument("--image-subdir", default="img1", help="Image folder under --video-dir.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <video-dir>/visualization_local.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_dir = Path(args.video_dir)
    video_name = video_dir.name
    image_folder = video_dir / args.image_subdir
    json_path = video_dir / f"{video_name}.json"
    output_dir = Path(args.output_dir) if args.output_dir else video_dir / "visualization_local"

    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if not image_folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    if not json_path.is_file():
        raise FileNotFoundError(f"Prediction JSON not found: {json_path}")

    print(f"Video directory : {video_dir}")
    print(f"Image folder    : {image_folder}")
    print(f"Prediction JSON : {json_path}")
    print(f"Output directory: {output_dir}")

    visualize_predictions(str(image_folder), str(output_dir), str(json_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
