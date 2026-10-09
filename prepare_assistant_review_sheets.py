"""Prepare timestamped evidence contact sheets for direct visual inspection."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def prepare(bundle_path, destination, event_id=None, indices=None):
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    for event in bundle["events"]:
        if event_id and event["id"] != event_id:
            continue
        frames = event["evidenceFrames"]
        selected = indices or sorted(set([*range(0, len(frames), 4), len(frames) - 1]))
        sheet = Image.new("RGB", (1920, 394 * math.ceil(len(selected) / 3)), "#13171b")
        draw = ImageDraw.Draw(sheet)
        for cell, index in enumerate(selected):
            frame = frames[index]
            x, y = (cell % 3) * 640, (cell // 3) * 394
            with Image.open(frame["path"]) as image:
                sheet.paste(image.resize((640, 360)), (x, y + 34))
            draw.text((x + 8, y + 3), f"{event['id']}  #{index:02d}  {frame['time']:.3f}s", font=font, fill="white")
        suffix = "_detail" if indices else ""
        sheet.save(destination / f"{event['id']}{suffix}.jpg", quality=93)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event")
    parser.add_argument("--indices", type=int, nargs="+")
    args = parser.parse_args()
    prepare(args.bundle, args.output_dir, args.event, args.indices)
