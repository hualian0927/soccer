"""Extract L3/L4 review windows without calling an external model."""

import argparse
import json
from pathlib import Path

from prepare_assistant_review_sheets import prepare
from review_tactical_candidates_with_vision import extract_frames, save_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    video = Path(bundle["source"]["video"])
    items = [dict(item, level=level) for level in ("L3", "L4") for item in bundle["layers"][level]["items"]]
    events = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        # Review the full proposed interval plus context, not only its first frame.
        start = max(0, item["time"] - 5)
        end = min(bundle["source"]["duration_sec"], max(item["time"], item.get("end", item["time"])) + 5)
        center = (start + end) / 2
        event = {"time_sec": center}
        frames = extract_frames(video, event, 21, 1280, center - start, end - center)
        saved = save_evidence(video, event, frames, args.output_dir / "evidence" / item["id"], center - start, end - center)
        events.append({**item, "evidenceStart": start, "evidenceEnd": end, "clipPath": saved["clip_path"],
                       "evidenceFrames": [{"path": p, "time": f["time_sec"]}
                                          for p, f in zip(saved["image_paths"], frames)]})
        print(item["id"], flush=True)
    manifest = args.output_dir / "review_evidence.json"
    manifest.write_text(json.dumps({"events": events}, ensure_ascii=False, indent=2), encoding="utf-8")
    prepare(manifest, args.output_dir / "contact_sheets", indices=list(range(0, 21, 2)))


if __name__ == "__main__":
    main()
