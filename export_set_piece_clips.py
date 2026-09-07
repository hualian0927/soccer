#!/usr/bin/env python3
"""Export one reviewable MP4 clip for every detected set-piece event."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


TYPE_LABELS = {
    "corner": "corner",
    "free_kick": "free-kick",
    "goal_kick": "goal-kick",
    "throw_in": "throw-in",
    "kickoff": "kickoff",
    "penalty": "penalty",
    "unknown": "set-piece",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export set-piece clips from the L1 manifest.")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-") or "event"


def main() -> int:
    args = parse_args()
    input_video = Path(args.input_video).resolve()
    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_video.exists():
        raise FileNotFoundError(input_video)
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    exported = []
    for index, row in enumerate(rows, 1):
        start = max(0.0, float(row["evidence_start_sec"]))
        end = max(start + 0.5, float(row["evidence_end_sec"]))
        subtype = row.get("event_subtype") or "unknown"
        filename = f"{index:03d}_{TYPE_LABELS.get(subtype, 'set-piece')}_{start:07.2f}s.mp4"
        output = output_dir / filename
        command = [
            args.ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(input_video),
            "-t", f"{end - start:.3f}", "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-movflags", "+faststart", str(output),
        ]
        subprocess.run(command, check=True)
        exported.append({
            **row,
            "clip_id": safe_id(row["event_id"]),
            "clip_path": str(output),
            "clip_filename": filename,
            "clip_duration_sec": round(end - start, 3),
        })
    index_path = output_dir / "set_piece_clips.json"
    index_path.write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(exported)} set-piece clips to {output_dir}")
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
