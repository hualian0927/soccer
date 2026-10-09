#!/usr/bin/env python3
"""Visualize referee-refined SoccerNetGSR results with a distinct referee color."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import yaml

from workflows.gsr.run_local_video_gsr_visualization import make_video_from_frames


TEMPLATE_IMAGE_PATH = "sfr/template/Radar.png"
TEAM_LEFT_COLOR = (0, 0, 255)       # Red, BGR
TEAM_RIGHT_COLOR = (255, 0, 0)      # Blue, BGR
BALL_COLOR = (0, 255, 255)          # Yellow, BGR
REFEREE_COLOR = (255, 0, 255)       # Magenta, BGR
DEFAULT_COLOR = (0, 255, 0)         # Green, BGR


def load_visual_config() -> dict:
    config_path = Path("soccer_input_dataset/gsr_demo/config.local.yaml")
    if not config_path.is_file():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("VISUALIZATION", {})


def format_image_name(image_id: str) -> str:
    return f"{int(str(image_id)[-6:]):06d}.jpg"


def meter_to_pixel(x_meter: float, y_meter: float, center_x_px: float, center_y_px: float) -> tuple[int, int]:
    return int(x_meter + center_x_px), int(y_meter + center_y_px)


def draw_prediction(image, radar_overlay, pred, center_x_px, center_y_px, player_radius, ball_radius):
    attrs = pred.get("attributes", {})
    role = str(attrs.get("role", "")).lower()
    track_id = pred.get("track_id")
    bbox_image = pred.get("bbox_image")
    bbox_pitch = pred.get("bbox_pitch")

    if role == "ball":
        label = "Ball"
        color = BALL_COLOR
        thickness = 3
        radius = ball_radius
    elif role == "referee":
        label = "Referee"
        color = REFEREE_COLOR
        thickness = 2
        radius = player_radius
    elif role == "other":
        label = "Other"
        color = DEFAULT_COLOR
        thickness = 2
        radius = player_radius
    else:
        jersey = attrs.get("jersey")
        jersey = "100" if jersey is None or jersey == "100" else jersey
        team = str(attrs.get("team", "")).lower()
        if team == "left":
            color = TEAM_LEFT_COLOR
            team_label = "L"
        elif team == "right":
            color = TEAM_RIGHT_COLOR
            team_label = "R"
        else:
            color = DEFAULT_COLOR
            team_label = ""
        label = f"ID:{track_id}, J:{jersey}, {team_label}"
        thickness = 2
        radius = player_radius

    if bbox_image:
        x = int(bbox_image["x"])
        y = int(bbox_image["y"])
        w = int(bbox_image["w"])
        h = int(bbox_image["h"])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if bbox_pitch:
        x_px, y_px = meter_to_pixel(
            bbox_pitch["x_bottom_middle"],
            bbox_pitch["y_bottom_middle"],
            center_x_px,
            center_y_px,
        )
        cv2.circle(radar_overlay, (x_px, y_px), radius, color, -1)


def visualize_predictions(image_folder: Path, json_path: Path, output_dir: Path) -> Path:
    radar_template = cv2.imread(TEMPLATE_IMAGE_PATH)
    if radar_template is None:
        raise FileNotFoundError(f"Radar template not found: {TEMPLATE_IMAGE_PATH}")

    visual_cfg = load_visual_config()
    player_radius = int(visual_cfg.get("PLAYER_RADAR_RADIUS", 2))
    ball_radius = int(visual_cfg.get("BALL_RADAR_RADIUS", player_radius))
    center_x_px = radar_template.shape[1] / 2
    center_y_px = radar_template.shape[0] / 2

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id_to_predictions = {}
    image_id_to_video_id = {}
    for pred in data.get("predictions", []):
        image_id = pred.get("image_id")
        video_id = pred.get("video_id")
        image_id_to_predictions.setdefault(image_id, []).append(pred)
        image_id_to_video_id[image_id] = video_id

    base_result_dir = output_dir / "Predict_Visualization"
    for image_id, predictions in image_id_to_predictions.items():
        file_name = format_image_name(image_id)
        image_path = image_folder / file_name
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Image not found: {image_path}")
            continue

        radar_overlay = radar_template.copy()
        for pred in predictions:
            draw_prediction(image, radar_overlay, pred, center_x_px, center_y_px, player_radius, ball_radius)

        radar_resized = cv2.resize(radar_overlay, (image.shape[1] // 4, image.shape[0] // 4))
        template_pos = (image.shape[1] - radar_resized.shape[1] - 10, 10)
        image[
            template_pos[1] : template_pos[1] + radar_resized.shape[0],
            template_pos[0] : template_pos[0] + radar_resized.shape[1],
        ] = radar_resized

        video_id = image_id_to_video_id.get(image_id, "unknown")
        result_dir = base_result_dir / f"SNGS-{video_id}"
        result_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(result_dir / file_name), image)

    video_name = f"SNGS-{next(iter(image_id_to_video_id.values()))}" if image_id_to_video_id else image_folder.parent.name
    return base_result_dir / video_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize referee-refined JSON with magenta referee boxes.")
    parser.add_argument("--video-dir", default="soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-999")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--output-dir", default="soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-999/visualization_referee_magenta")
    parser.add_argument("--output-video", default="soccer_input_dataset/outputs/test_gsr_visualized_full_referee_magenta.mp4")
    parser.add_argument("--fps", type=float, default=25.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_dir = Path(args.video_dir)
    json_path = Path(args.json_path) if args.json_path else video_dir / f"{video_dir.name}.referee_refined.json"
    output_dir = Path(args.output_dir)
    frames_dir = visualize_predictions(video_dir / "img1", json_path, output_dir)
    make_video_from_frames(frames_dir, Path(args.output_video), args.fps)
    print(f"Wrote video: {args.output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
