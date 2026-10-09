#!/usr/bin/env python3
"""Export ranked tactical highlight clips from a generated manifest."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--minimum-score", type=float, default=0.70)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_highlights(path: Path, top_n: int, minimum_score: float) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if float(row["score"]) >= minimum_score]
    selected.sort(key=lambda row: (-float(row["score"]), float(row["center_sec"])))
    return selected[: max(0, top_n)]


def main() -> int:
    args = parse_args()
    if not args.input_video.exists():
        raise FileNotFoundError(args.input_video)
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and not args.dry_run:
        raise RuntimeError("ffmpeg is required; activate the sports conda environment first")

    highlights = load_highlights(args.manifest, args.top_n, args.minimum_score)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for rank, item in enumerate(highlights, 1):
        start = float(item["start_sec"])
        duration = max(0.2, float(item["duration_sec"]))
        event_type = str(item["type"]).replace("_candidate", "").replace("_", "-")
        output = args.output_dir / f"{rank:02d}_{event_type}_{start:07.2f}s.mp4"
        command = [
            ffmpeg or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(args.input_video),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ]
        if args.dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, check=True)
        exported.append(
            {
                "rank": rank,
                "highlight_id": item["highlight_id"],
                "type": item["type"],
                "label_zh": item["label_zh"],
                "score": float(item["score"]),
                "start_sec": start,
                "duration_sec": duration,
                "output": str(output),
            }
        )
        print(f"[{rank}/{len(highlights)}] {output}")
    index_path = args.output_dir / "exported_highlights.json"
    index_path.write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Export index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
