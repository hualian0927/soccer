#!/usr/bin/env python3
"""Run the SoccerNetGSR pipeline on a local mp4 and export a visualization video."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import yaml

from kpts import predict as predict_homography


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a visualized SoccerNetGSR demo video from a local mp4.")
    parser.add_argument("--input-video", default="soccer_input_dataset/test.mp4")
    parser.add_argument("--video-name", default="SNGS-999")
    parser.add_argument("--start-frame", type=int, default=1, help="1-based input frame to start extraction from.")
    parser.add_argument("--max-frames", type=int, default=120, help="Use 0 to process all frames.")
    parser.add_argument("--output-video", default="soccer_input_dataset/outputs/test_gsr_visualized.mp4")
    parser.add_argument("--work-root", default="soccer_input_dataset/gsr_demo")
    parser.add_argument("--batch-size", type=int, default=4, help="Frame batch size for detector inference.")
    parser.add_argument(
        "--homography-stride",
        type=int,
        default=1,
        help="Recompute pitch homography every N frames and reuse it between keyframes.",
    )
    parser.add_argument(
        "--recall-optimized",
        action="store_true",
        help="Use lower detector/tracker thresholds and softer final filtering for demo recall.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Skip slow CLIP jersey recognition and use lightweight geometry/color heuristics for long-video previews.",
    )
    parser.add_argument("--team0-colors", default="", help="Comma-separated colors for team 0, e.g. blue,yellowgreen.")
    parser.add_argument("--team1-colors", default="", help="Comma-separated colors for team 1, e.g. white.")
    parser.add_argument("--referee-colors", default="", help="Comma-separated referee colors, e.g. red.")
    parser.add_argument("--goalkeeper-team0-colors", default="", help="Optional comma-separated goalkeeper colors for team 0.")
    parser.add_argument("--goalkeeper-team1-colors", default="", help="Optional comma-separated goalkeeper colors for team 1.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_frames(input_video: Path, image_dir: Path, max_frames: int, start_frame: int) -> float:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start_frame > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    image_dir.mkdir(parents=True, exist_ok=True)

    frame_id = 1
    while True:
        if max_frames and frame_id > max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(image_dir / f"{frame_id:06d}.jpg"), frame)
        frame_id += 1
    cap.release()

    if frame_id == 1:
        raise RuntimeError(f"No frames extracted from {input_video}")
    print(f"Extracted {frame_id - 1} frames from input frame {start_frame} to {image_dir}")
    return fps


def split_colors(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_demo_config(
    work_root: Path,
    video_name: str,
    recall_optimized: bool,
    fast_mode: bool,
    batch_size: int,
    team0_colors: list[str] | None = None,
    team1_colors: list[str] | None = None,
    referee_colors: list[str] | None = None,
    goalkeeper_team0_colors: list[str] | None = None,
    goalkeeper_team1_colors: list[str] | None = None,
) -> Path:
    with open(PROJECT_ROOT / "configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["DATA_DIR"] = str(work_root / "SoccerNetGS")
    cfg["DATA_SETS"] = ["test"]
    cfg["TARGET_VIDEO_NAME"] = video_name
    cfg["JERSEY_MODE"] = "CLIP"
    cfg["FAST_MODE"] = bool(fast_mode)
    cfg["INFERENCE_BATCH_SIZE"] = int(batch_size)
    color_hints = {
        "TEAM0_COLORS": team0_colors or [],
        "TEAM1_COLORS": team1_colors or [],
        "REFEREE_COLORS": referee_colors or [],
        "GOALKEEPER_TEAM0_COLORS": goalkeeper_team0_colors or [],
        "GOALKEEPER_TEAM1_COLORS": goalkeeper_team1_colors or [],
    }
    if any(color_hints.values()):
        cfg["COLOR_HINTS"] = color_hints
        cfg["TEAM_IDENTITY"] = {
            "MIN_EVIDENCE_FRAMES": 3,
            "MIN_CONFIDENCE": 0.58,
            "MIN_MARGIN": 0.18,
        }

    if fast_mode:
        cfg.setdefault("TRACKER", {})["WITH_REID"] = False
        cfg["MAX_REFEREES_PER_FRAME"] = 3

    if recall_optimized:
        tracker_cfg = cfg.setdefault("TRACKER", {})
        tracker_cfg["CONF"] = 0.03
        tracker_cfg["TRACK_HIGH_THRESH"] = 0.30
        tracker_cfg["TRACK_LOW_THRESH"] = 0.01
        tracker_cfg["NEW_TRACK_THRESH"] = 0.30
        tracker_cfg["TRACK_BUFFER"] = 90
        tracker_cfg["MIN_BOX_AREA"] = 2
        tracker_cfg["ASPECT_RATIO_THRESH"] = 3.2
        cfg["POSTPROCESS"] = {
            "BALL_MAX_AREA": 500,
            "BALL_MAX_ASPECT": 1.5,
            "BALL_MAX_HEIGHT": 20,
            "BALL_MAX_WIDTH": 20,
            "BALL_MIN_FRAMES": 10,
            "BALL_MIN_VOTE_RATIO": 0.6,
            "BALL_USE_OPTICAL_FLOW": True,
            "BALL_MIN_FLOW_VALID_RATIO": 0.7,
            "BALL_FLOW_MAX_PAIRS": 120,
            "BALL_FLOW_MAX_RESIDUAL": 6.0,
            "BALL_FLOW_MAX_ERROR": 30.0,
            "MIN_FRAME_THRESHOLD": 8,
        }
        cfg["VISUALIZATION"] = {
            "PLAYER_RADAR_RADIUS": 2,
            "BALL_RADAR_RADIUS": 2,
            "BALL_BOX_THICKNESS": 3,
        }

    config_path = work_root / "config.local.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return config_path


def run_step(label: str, args: list[str]) -> None:
    print(f"\n[{label}] {' '.join(args)}")
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def copy_homographies(result_dir: Path, image_dir: Path) -> int:
    npy_dir = result_dir / "npy_files"
    npy_files = sorted(npy_dir.glob("*.npy"))
    if not npy_files:
        existing = sorted(image_dir.glob("*.npy"))
        if existing:
            print(f"Homography files already available in {image_dir}: {len(existing)}")
            return len(existing)
        raise RuntimeError(f"No homography .npy files found in {npy_dir} or {image_dir}")
    for npy_path in npy_files:
        shutil.copy2(npy_path, image_dir / npy_path.name)
    print(f"Copied {len(npy_files)} homography files to {image_dir}")
    return len(npy_files)


def make_video_from_frames(frame_dir: Path, output_video: Path, fps: float) -> None:
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No visualization frames found in {frame_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read first visualization frame: {frames[0]}")

    height, width = first.shape[:2]
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")

    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    print(f"Wrote visualization video: {output_video}")


def main() -> int:
    args = parse_args()
    input_video = (PROJECT_ROOT / args.input_video).resolve()
    work_root = (PROJECT_ROOT / args.work_root).resolve()
    output_video = (PROJECT_ROOT / args.output_video).resolve()
    video_dir = work_root / "SoccerNetGS" / "test" / args.video_name
    image_dir = video_dir / "img1"
    result_dir = work_root / "Results" / args.video_name

    if video_dir.exists() and args.overwrite:
        shutil.rmtree(video_dir)
    if result_dir.exists() and args.overwrite:
        shutil.rmtree(result_dir)

    if not image_dir.exists() or not any(image_dir.glob("*.jpg")):
        fps = extract_frames(input_video, image_dir, args.max_frames, max(1, args.start_frame))
    else:
        cap = cv2.VideoCapture(str(input_video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        print(f"Reusing existing frames in {image_dir}")

    config_path = write_demo_config(
        work_root,
        args.video_name,
        args.recall_optimized,
        args.fast_mode,
        args.batch_size,
        team0_colors=split_colors(args.team0_colors),
        team1_colors=split_colors(args.team1_colors),
        referee_colors=split_colors(args.referee_colors),
        goalkeeper_team0_colors=split_colors(args.goalkeeper_team0_colors),
        goalkeeper_team1_colors=split_colors(args.goalkeeper_team1_colors),
    )

    image_count = len(list(image_dir.glob("*.jpg")))
    npy_count = len(list(image_dir.glob("*.npy")))
    if npy_count >= image_count and image_count > 0:
        print(f"\n[homography] Reusing {npy_count} existing homography files in {image_dir}")
    else:
        print("\n[homography] kpts.predict")
        predict_homography(
            img_path=str(image_dir),
            result_dir=str(result_dir),
            checkpoints="checkpoints/SoccernetGSR_EfficientNet_Best.pth",
            template_image="template/Radar_Dimen.png",
            template_npy="template/soccernet_template_97.npy",
            save_viz=False,
            verbose=True,
            frame_skip=max(1, args.homography_stride),
        )
    copy_homographies(result_dir, image_dir)

    run_step("tracking-recognition", [sys.executable, "inference_soccernetGSR.py", "--config", str(config_path)])
    run_step("remove-duplicates", [sys.executable, "IDATR/rmv_doub_bbox.py", "--config", str(config_path)])
    if args.fast_mode:
        # ReID is disabled in fast mode, so its embeddings are zero vectors and
        # appearance-based all-pairs tracklet merging has no valid evidence.
        rmved_path = video_dir / f"rmved_{args.video_name}.txt"
        refined_path = video_dir / f"refined_{args.video_name}.txt"
        shutil.copy2(rmved_path, refined_path)
        print(f"Fast mode: reused de-duplicated tracks as {refined_path}")
    else:
        run_step("generate-tracklets", [sys.executable, "IDATR/gen_tracklets.py", "--config", str(config_path)])
        run_step("refine-tracklets", [sys.executable, "IDATR/refine_tracklets.py", "--config", str(config_path)])
    run_step("refine-team-identities", [sys.executable, "refine_team_identities.py", "--config", str(config_path)])
    run_step("court-meter", [sys.executable, "IDATR/create_court_file.py", "--config", str(config_path)])
    run_step("json-format", [sys.executable, "write_json_file_team.py", "--config", str(config_path)])

    visualization_dir = video_dir / "visualization_local_video"
    run_step(
        "visualize-json",
        [
            sys.executable,
            "visualize_local_result.py",
            "--video-dir",
            str(video_dir),
            "--output-dir",
            str(visualization_dir),
        ],
    )

    video_id = args.video_name.split("-")[-1]
    visual_frame_dir = visualization_dir / "Predict_Visualization" / f"SNGS-{video_id}"
    make_video_from_frames(visual_frame_dir, output_video, fps)

    print(f"\nDone: {output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
