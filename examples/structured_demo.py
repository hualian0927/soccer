"""Run synthetic tracking through the tactical pipeline without weights or video."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tactical_analysis.pipeline import TacticalAnalysisPipeline
from tactical_analysis.reporting import write_report_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "soccer_input_dataset/outputs/synthetic_demo")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    shape = [(-35,y) for y in (-24,-8,8,24)] + [(-10,y) for y in (-18,0,18)] + [(15,y) for y in (-18,0,18)]
    for frame in range(1,76):
        entries = [(index+offset,"player",team,x*direction,y)
                   for team,offset,direction in (("left",1,1),("right",101,-1))
                   for index,(x,y) in enumerate(shape)]
        entries += [(90,"goalkeeper","left",-49,0),(190,"goalkeeper","right",49,0),(999,"ball",None,-10+frame*.12,0)]
        for track,role,team,x,y in entries:
            predictions.append({"image_id":f"9000{frame:06d}","track_id":track,
                "bbox_pitch":{"x_bottom_middle":x,"y_bottom_middle":y},
                "bbox_image":{"x":x+100,"y":y+100,"w":20,"h":50},
                "attributes":{"role":role,"team":team,"jersey":None}})
    source = args.output_dir / "synthetic_gsr.json"
    source.write_text(json.dumps({"synthetic":True,"predictions":predictions}),encoding="utf-8")
    pipeline = TacticalAnalysisPipeline.from_config_file(ROOT / "configs/tactical_analysis.yaml")
    pipeline.config["pipeline"]["continue_on_error"] = False
    report = pipeline.run(source, fps=25)
    write_report_bundle(report,args.output_dir)
    print(f"Synthetic smoke test only, not real-match evidence: {args.output_dir}")


if __name__ == "__main__":
    main()
