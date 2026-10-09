"""Apply the existing kit-color tracklet classifier to an exported GSR JSON."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import pandas as pd

from workflows.identity.jersey_color import estimate_jersey_color
from workflows.identity.team_identity import apply_track_identities, infer_track_identities, split_identity_inconsistent_tracks
from tactical_analysis.gsr_io import parse_frame


def refine(video: Path, source: Path, output: Path, color_groups: dict[str, set[str]], stride: int = 3) -> dict:
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for index, prediction in enumerate(data.get("predictions", [])):
        attrs = prediction.get("attributes") or {}
        box = prediction.get("bbox_image") or {}
        rows.append({
            "index": index, "frame": parse_frame(prediction.get("image_id")),
            "track_id": int(prediction.get("track_id", -1)), "role": attrs.get("role", "player"),
            "team": attrs.get("team"), "color": "unknown", "score": 0.8,
            **{key: float(box.get(key, 0)) for key in ("x", "y", "w", "h")},
        })
    table = pd.DataFrame(rows).set_index("index")
    people = table[table.role.str.lower() != "ball"]
    by_frame = defaultdict(list)
    for index, row in people.iterrows():
        if int(row.frame) % stride == 0:
            by_frame[int(row.frame)].append((index, row))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot read video: {video}")
    frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id += 1
            for index, row in by_frame.get(frame_id, []):
                x1, y1 = max(0, int(row.x)), max(0, int(row.y))
                x2 = min(frame.shape[1], int(row.x + row.w))
                y2 = min(frame.shape[0], int(row.y + row.h))
                if x2 > x1 and y2 > y1:
                    estimate = estimate_jersey_color(frame[y1:y2, x1:x2])
                    table.at[index, "color"] = estimate.label
            if frame_id % 1800 == 0:
                print(f"Kit evidence: {frame_id} frames", flush=True)
    finally:
        capture.release()
    split, segments = split_identity_inconsistent_tracks(table, color_groups)
    decisions = infer_track_identities(split, color_groups)
    # Officials and goalkeepers need repeated kit observations, not one bright crop.
    decisions = {key: value for key, value in decisions.items()
                 if value.identity in {"team0", "team1"} or value.evidence_frames >= 3}
    classified = apply_track_identities(split, decisions)
    for index, row in classified.iterrows():
        prediction = data["predictions"][int(index)]
        if str(row.role).lower() == "ball":
            continue
        attrs = prediction.setdefault("attributes", {})
        prediction["source_track_id"] = prediction["track_id"]
        prediction["track_id"] = int(row.track_id)
        attrs["role"] = str(row.role).lower()
        attrs["team"] = {0: "left", 1: "right"}.get(row.team)
        identity = decisions.get(int(row.track_id))
        attrs["identity_source"] = "kit_tracklet_vote" if identity else "unresolved"
        attrs["identity_confidence"] = identity.confidence if identity else 0.0
    data["identity_refinement"] = {
        "source_json": str(source.resolve()), "source_video": str(video.resolve()),
        "color_groups": {key: sorted(value) for key, value in color_groups.items()},
        "sample_stride": stride, "split_segments": len(segments),
        "role_counts": dict(Counter(p.get("attributes", {}).get("role") for p in data["predictions"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data["identity_refinement"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--team0-colors", default="blue")
    parser.add_argument("--team1-colors", default="white")
    parser.add_argument("--referee-colors", default="red")
    parser.add_argument("--goalkeeper-team0-colors", default="yellow,green")
    parser.add_argument("--goalkeeper-team1-colors", default="")
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    if args.stride < 1 or args.stride > 5:
        parser.error("stride must be between 1 and 5")
    if args.json_path.resolve() == args.output_json.resolve():
        parser.error("use a separate output JSON")
    groups = {name: {part.strip() for part in value.split(",") if part.strip()} for name, value in {
        "team0": args.team0_colors, "team1": args.team1_colors,
        "referee": args.referee_colors, "goalkeeper_team0": args.goalkeeper_team0_colors,
        "goalkeeper_team1": args.goalkeeper_team1_colors,
    }.items()}
    print(json.dumps(refine(args.video, args.json_path, args.output_json, groups, args.stride), ensure_ascii=False))


if __name__ == "__main__":
    main()
