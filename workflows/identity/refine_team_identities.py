#!/usr/bin/env python3
"""Re-evaluate team identities after IDATR changes track IDs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import pandas as pd
import yaml

from workflows.identity.jersey_color import estimate_jersey_color
from workflows.identity.team_identity import (
    apply_track_identities,
    identity_report_rows,
    infer_track_identities,
    split_identity_inconsistent_tracks,
)
from workflows.identity.tracklet_appearance import refine_identities_with_clip


COLUMNS = ["frame", "track_id", "x", "y", "w", "h", "score", "role", "jersey", "color", "team"]
COLOR_ALIASES = {
    "蓝": "blue", "蓝色": "blue", "白": "white", "白色": "white",
    "红": "red", "红色": "red", "黄": "yellow", "黄色": "yellow",
    "绿": "green", "绿色": "green", "黄绿": "yellow", "黄绿色": "yellow",
    "黑": "black", "黑色": "black", "灰": "grey", "灰色": "grey",
    "橙": "orange", "橙色": "orange", "紫": "purple", "紫色": "purple",
}


def colors(value) -> set[str]:
    if not value:
        return set()
    values = value if isinstance(value, list) else str(value).split(",")
    result = set()
    for raw in values:
        token = str(raw).strip().lower().replace(" ", "")
        if not token:
            continue
        if token in {"yellowgreen", "yellow-green", "yellow_green"}:
            result.update({"yellow", "green"})
        else:
            result.add(COLOR_ALIASES.get(token, token))
    return result


def classify_rows(frame: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    result = frame.copy()
    for frame_id, rows in result.groupby("frame", sort=False):
        image = cv2.imread(str(image_dir / f"{int(frame_id):06d}.jpg"))
        if image is None:
            continue
        height, width = image.shape[:2]
        for index, row in rows.iterrows():
            if str(row["role"]).strip().lower() == "ball":
                continue
            x1 = max(0, int(float(row["x"])))
            y1 = max(0, int(float(row["y"])))
            x2 = min(width, int(float(row["x"]) + float(row["w"])))
            y2 = min(height, int(float(row["y"]) + float(row["h"])))
            crop = image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
            result.at[index, "color"] = estimate_jersey_color(crop).label
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    video_name = cfg.get("TARGET_VIDEO_NAME")
    if not video_name:
        raise ValueError("TARGET_VIDEO_NAME is required")
    video_dir = Path(cfg["DATA_DIR"]) / "test" / video_name
    refined_path = video_dir / f"refined_{video_name}.txt"
    image_dir = video_dir / "img1"
    if not refined_path.exists():
        raise FileNotFoundError(refined_path)
    backup_path = video_dir / f"refined_before_team_identity_{video_name}.txt"
    source_path = backup_path if backup_path.exists() else refined_path

    color_hints = cfg.get("COLOR_HINTS") or {}
    if not any(color_hints.values()):
        print("No COLOR_HINTS configured; post-IDATR team refinement skipped.")
        return 0
    frame = pd.read_csv(source_path, header=None, names=COLUMNS)
    frame = classify_rows(frame, image_dir)
    team0 = colors(color_hints.get("TEAM0_COLORS"))
    team1 = colors(color_hints.get("TEAM1_COLORS"))
    referee = colors(color_hints.get("REFEREE_COLORS"))
    gk0 = colors(color_hints.get("GOALKEEPER_TEAM0_COLORS"))
    gk1 = colors(color_hints.get("GOALKEEPER_TEAM1_COLORS"))
    identity_cfg = cfg.get("TEAM_IDENTITY") or {}
    color_groups = {
        "team0": team0 - gk0,
        "team1": team1 - gk1,
        "goalkeeper_team0": gk0,
        "goalkeeper_team1": gk1,
        "referee": referee,
    }
    split_cfg = cfg.get("TRACKLET_IDENTITY_SPLIT") or {}
    original_track_count = int(frame.track_id.nunique())
    frame, split_records = split_identity_inconsistent_tracks(
        frame,
        color_groups,
        maximum_gap_frames=int(split_cfg.get("MAXIMUM_GAP_FRAMES", 15)),
        smoothing_window=int(split_cfg.get("SMOOTHING_WINDOW", 13)),
        minimum_switch_run=int(split_cfg.get("MINIMUM_SWITCH_RUN", 10)),
        minimum_votes=int(split_cfg.get("MINIMUM_VOTES", 3)),
        minimum_ratio=float(split_cfg.get("MINIMUM_RATIO", 0.62)),
    )
    decisions = infer_track_identities(
        frame,
        color_groups,
        minimum_evidence_frames=int(identity_cfg.get("MIN_EVIDENCE_FRAMES", 3)),
        minimum_confidence=float(identity_cfg.get("MIN_CONFIDENCE", 0.58)),
        minimum_margin=float(identity_cfg.get("MIN_MARGIN", 0.18)),
    )
    appearance_cfg = cfg.get("APPEARANCE_REFINEMENT") or {}
    appearance_report = {"status": "disabled"}
    if bool(appearance_cfg.get("ENABLED", False)):
        decisions, appearance_report = refine_identities_with_clip(
            frame,
            image_dir,
            decisions,
            color_groups,
            model_path=Path(appearance_cfg.get("MODEL_PATH", "checkpoints/CLIP_Jersey.pth")).resolve(),
            maximum_samples_per_track=int(appearance_cfg.get("SAMPLES_PER_TRACK", 6)),
            batch_size=int(appearance_cfg.get("BATCH_SIZE", 16)),
            minimum_margin=float(appearance_cfg.get("MINIMUM_MARGIN", 0.06)),
            override_color_confidence_below=float(
                appearance_cfg.get("OVERRIDE_COLOR_CONFIDENCE_BELOW", 0.86)
            ),
            maximum_anchor_tracks_per_team=int(
                appearance_cfg.get("ANCHOR_TRACKS_PER_TEAM", 40)
            ),
        )
    refined = apply_track_identities(
        frame,
        decisions,
        max_referees_per_frame=int(cfg.get("MAX_REFEREES_PER_FRAME", 3)),
    )
    if not backup_path.exists():
        shutil.copy2(source_path, backup_path)
    refined.to_csv(refined_path, index=False, header=False)
    report = {
        "stage": "post_idatr",
        "video_name": video_name,
        "source": str(source_path),
        "original_track_count": original_track_count,
        "identity_consistent_track_count": int(frame.track_id.nunique()),
        "split_segment_count": len(split_records),
        "stable_track_count": len(decisions),
        "unresolved_track_count": int(frame.track_id.nunique() - len(decisions)),
        "tracklet_splits": split_records,
        "appearance_refinement": appearance_report,
        "tracks": identity_report_rows(decisions.values()),
    }
    report_path = video_dir / f"team_identity_{video_name}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Post-IDATR team refinement: stable={report['stable_track_count']}, "
        f"unresolved={report['unresolved_track_count']}"
    )
    print(f"Saved: {refined_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
