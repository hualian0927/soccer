from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tactical_analysis.pipeline import TacticalAnalysisPipeline
from tactical_analysis.reporting import write_report_bundle


def prediction(frame: int, track_id: int, role: str, team: str | None, x: float, y: float) -> dict:
    return {
        "image_id": f"9000{frame:06d}",
        "track_id": track_id,
        "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y},
        "bbox_image": {"x": x + 100, "y": y + 100, "w": 20, "h": 50},
        "attributes": {"role": role, "team": team, "jersey": None},
    }


class TacticalAnalysisFrameworkTest(unittest.TestCase):
    def test_pipeline_writes_all_report_formats(self) -> None:
        predictions = []
        left_shape = [(-38, -24), (-36, -8), (-34, 8), (-32, 24), (-10, -16), (-8, 0), (-6, 16), (16, 0)]
        right_shape = [(38, -24), (36, -8), (34, 8), (32, 24), (10, -16), (8, 0), (6, 16), (-16, 0)]
        for frame in range(1, 76):
            for index, (x, y) in enumerate(left_shape, 1):
                predictions.append(prediction(frame, index, "player", "left", x + frame * 0.02, y))
            for index, (x, y) in enumerate(right_shape, 101):
                predictions.append(prediction(frame, index, "player", "right", x - frame * 0.02, y))
            predictions.append(prediction(frame, 90, "goalkeeper", "left", -49, 0))
            predictions.append(prediction(frame, 190, "goalkeeper", "right", 49, 0))
            ball_x = -10 + frame * 0.12
            predictions.append(prediction(frame, 999, "ball", None, ball_x, 0))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "sample.json"
            json_path.write_text(json.dumps({"predictions": predictions}), encoding="utf-8")
            pipeline = TacticalAnalysisPipeline(
                {
                    "analyzers": {
                        "formation_tendency": {"sample_step_frames": 5, "min_visible_players": 7},
                        "relational_space": {"sample_step_frames": 5},
                    }
                }
            )
            report = pipeline.run(json_path=json_path, fps=25.0)
            analyzer_names = {item.analyzer for item in report.analyzer_outputs}
            self.assertEqual(len(report.analyzer_outputs), 18)
            self.assertTrue(
                {
                    "data_quality",
                    "event_timeline",
                    "defensive_interventions",
                    "spatial_structure",
                    "team_shape_engine",
                    "formation_tendency",
                    "set_piece_delivery",
                    "goalkeeper_interventions",
                    "progression_analysis",
                    "attacking_chains",
                    "defensive_third_risk",
                    "pass_network",
                    "numerical_superiority",
                    "defensive_gaps",
                }.issubset(analyzer_names)
            )
            self.assertFalse(
                any(
                    warning.startswith("分析器运行失败")
                    for output in report.analyzer_outputs
                    for warning in output.warnings
                )
            )
            paths = write_report_bundle(report, root / "output")
            self.assertTrue(all(path.exists() for path in paths.values()))
            self.assertEqual(
                set(paths),
                {
                    "json", "markdown", "timeline", "summary", "highlights",
                    "l1_timeline_json", "l1_timeline_csv", "l1_summary", "set_piece_manifest",
                },
            )
            l1_catalog = json.loads(paths["l1_timeline_json"].read_text(encoding="utf-8"))
            self.assertEqual(l1_catalog["schema_version"], "1.0.0")
            self.assertIn("group_counts", l1_catalog["summary"])
            saved = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], "1.5.0")
            self.assertEqual(saved["provenance"]["visual_scope"], "video_tracking_only_no_gps_or_wearables")


if __name__ == "__main__":
    unittest.main()
