#!/usr/bin/env python3
"""Refine referee roles without modifying the current SoccerNetGSR pipeline.

This post-processes an existing SoccerNetGSR JSON using the per-frame source
votes in refined_SNGS-*.txt. It writes a new JSON and, optionally, a separate
visualized video.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from workflows.gsr.run_local_video_gsr_visualization import make_video_from_frames
from workflows.visualization.visualize_prediction_results import visualize_predictions


REFINED_COLUMNS = [
    "frame",
    "track_id",
    "x",
    "y",
    "w",
    "h",
    "score",
    "role",
    "jersey",
    "color",
    "team",
]


def parse_color_set(value: str) -> set[str]:
    return {item.strip().lower().replace("_", " ") for item in value.split(",") if item.strip()}


def majority_by_track(predictions: list[dict], key: str) -> dict[int, object]:
    votes = defaultdict(Counter)
    for pred in predictions:
        attrs = pred.get("attributes", {})
        votes[int(pred["track_id"])][attrs.get(key)] += 1
    return {track_id: counter.most_common(1)[0][0] for track_id, counter in votes.items() if counter}


def summarize_track(df: pd.DataFrame, track_id: int) -> dict:
    rows = df[df["track_id"] == track_id]
    roles = Counter(str(value).strip() for value in rows["role"])
    colors = Counter(str(value).strip().lower() for value in rows["color"])
    frames = len(rows)
    referee_votes = roles.get("Referee", 0)
    dominant_color, dominant_color_count = colors.most_common(1)[0]
    return {
        "track_id": track_id,
        "frames": frames,
        "roles": roles,
        "colors": colors,
        "referee_votes": referee_votes,
        "referee_ratio": referee_votes / frames if frames else 0.0,
        "dominant_color": dominant_color,
        "dominant_color_ratio": dominant_color_count / frames if frames else 0.0,
    }


def find_referee_tracks(
    predictions: list[dict],
    refined_df: pd.DataFrame,
    official_colors: set[str],
    min_frames: int,
    min_referee_votes: int,
    min_referee_ratio: float,
    min_official_color_ratio: float,
) -> tuple[set[int], list[dict]]:
    role_by_track = majority_by_track(predictions, "role")
    team_by_track = majority_by_track(predictions, "team")
    candidate_tracks = set()
    report_rows = []

    for track_id in sorted(refined_df["track_id"].dropna().astype(int).unique()):
        if role_by_track.get(track_id) != "player":
            continue
        summary = summarize_track(refined_df, track_id)
        official_color_votes = sum(summary["colors"].get(color, 0) for color in official_colors)
        official_color_ratio = official_color_votes / summary["frames"] if summary["frames"] else 0.0

        has_referee_votes = (
            summary["referee_votes"] >= min_referee_votes
            and summary["referee_ratio"] >= min_referee_ratio
        )
        has_official_kit = (
            summary["dominant_color"] in official_colors
            and official_color_ratio >= min_official_color_ratio
            and summary["referee_votes"] >= min_referee_votes
        )
        is_referee = summary["frames"] >= min_frames and (has_referee_votes or has_official_kit)

        row = {
            "track_id": track_id,
            "frames": summary["frames"],
            "json_team": team_by_track.get(track_id),
            "dominant_color": summary["dominant_color"],
            "dominant_color_ratio": round(summary["dominant_color_ratio"], 3),
            "official_color_ratio": round(official_color_ratio, 3),
            "referee_votes": summary["referee_votes"],
            "referee_ratio": round(summary["referee_ratio"], 3),
            "roles": summary["roles"].most_common(4),
            "colors": summary["colors"].most_common(6),
            "selected": is_referee,
        }
        report_rows.append(row)
        if is_referee:
            candidate_tracks.add(int(track_id))

    return candidate_tracks, report_rows


def rewrite_predictions(predictions: list[dict], referee_tracks: set[int]) -> list[dict]:
    refined = []
    for pred in predictions:
        new_pred = json.loads(json.dumps(pred))
        if int(new_pred["track_id"]) in referee_tracks:
            attrs = new_pred.setdefault("attributes", {})
            attrs["role"] = "referee"
            attrs["jersey"] = None
            attrs["team"] = None
            attrs["color"] = None
        refined.append(new_pred)
    return refined


def write_report(path: Path, referee_tracks: set[int], report_rows: list[dict]) -> None:
    lines = [
        "# Referee Refinement Report",
        "",
        f"Selected referee tracks: `{[int(track_id) for track_id in sorted(referee_tracks)]}`",
        "",
        "| track_id | selected | frames | team | dom_color | official_ratio | referee_votes | referee_ratio | roles | colors |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report_rows:
        if row["selected"] or row["referee_votes"] or row["official_color_ratio"] >= 0.15:
            lines.append(
                "| {track_id} | {selected} | {frames} | {json_team} | {dominant_color} | "
                "{official_color_ratio} | {referee_votes} | {referee_ratio} | `{roles}` | `{colors}` |".format(**row)
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a referee-refined SoccerNetGSR JSON/video.")
    parser.add_argument("--video-dir", default="soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-999")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--refined-path", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--official-colors", default="red,orange,yellow,neon yellow")
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-referee-votes", type=int, default=8)
    parser.add_argument("--min-referee-ratio", type=float, default=0.10)
    parser.add_argument("--min-official-color-ratio", type=float, default=0.60)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_dir = Path(args.video_dir)
    video_name = video_dir.name
    json_path = Path(args.json_path) if args.json_path else video_dir / f"{video_name}.json"
    refined_path = Path(args.refined_path) if args.refined_path else video_dir / f"refined_{video_name}.txt"
    output_json = Path(args.output_json) if args.output_json else video_dir / f"{video_name}.referee_refined.json"
    report_path = Path(args.report_path) if args.report_path else video_dir / f"{video_name}.referee_refined_report.md"
    output_dir = Path(args.output_dir) if args.output_dir else video_dir / "visualization_referee_refined"
    output_video = (
        Path(args.output_video)
        if args.output_video
        else Path("soccer_input_dataset/outputs") / "test_gsr_visualized_full_referee_refined.mp4"
    )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    refined_df = pd.read_csv(refined_path, header=None, names=REFINED_COLUMNS)
    refined_df["track_id"] = pd.to_numeric(refined_df["track_id"], errors="coerce")

    referee_tracks, report_rows = find_referee_tracks(
        data.get("predictions", []),
        refined_df,
        parse_color_set(args.official_colors),
        args.min_frames,
        args.min_referee_votes,
        args.min_referee_ratio,
        args.min_official_color_ratio,
    )
    new_predictions = rewrite_predictions(data.get("predictions", []), referee_tracks)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"predictions": new_predictions}, indent=4), encoding="utf-8")
    write_report(report_path, referee_tracks, report_rows)

    print(f"Referee tracks: {[int(track_id) for track_id in sorted(referee_tracks)]}")
    print(f"Wrote JSON: {output_json}")
    print(f"Wrote report: {report_path}")

    if args.visualize:
        image_dir = video_dir / "img1"
        visualize_predictions(str(image_dir), str(output_dir), str(output_json))
        frames_dir = output_dir / "Predict_Visualization" / video_name
        make_video_from_frames(frames_dir, output_video, args.fps)
        print(f"Wrote video: {output_video}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
