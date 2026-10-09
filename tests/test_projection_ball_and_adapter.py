import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from analyze_projection_with_ball import boundary_side, center_restart_candidates, outside_candidates
from ball_detection_adapter import merge_ball_trajectory


class OutOfBoundsEvidenceTests(unittest.TestCase):
    def test_line_type_and_calibration_margin(self):
        self.assertIsNone(boundary_side(-0.5, 20, 65, 101))
        self.assertEqual(boundary_side(-1.1, 20, 65, 101), "touchline")
        self.assertEqual(boundary_side(30, 102.2, 65, 101), "goal_line")

    def test_three_observed_consecutive_frames_required(self):
        rows = [
            {"frame": 1, "x_m": 10, "y_m": 50, "observed": True, "confidence": 0.9},
            {"frame": 2, "x_m": -1.2, "y_m": 50, "observed": True, "confidence": 0.9},
            {"frame": 3, "x_m": -1.3, "y_m": 50, "observed": False, "confidence": 0},
            {"frame": 4, "x_m": -1.2, "y_m": 50, "observed": True, "confidence": 0.9},
            {"frame": 5, "x_m": -1.4, "y_m": 50, "observed": True, "confidence": 0.9},
            {"frame": 6, "x_m": -1.5, "y_m": 50, "observed": True, "confidence": 0.9},
        ]
        candidates = outside_candidates(rows, 65, 101, 25)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["line"], "touchline")
        self.assertEqual(candidates[0]["status"], "needs_restart_review")

    def test_stationary_center_then_departure_is_only_candidate(self):
        rows = [
            {"frame": frame, "time_s": (frame - 1) / 25, "x_m": 32.5,
             "y_m": 50.5 if frame <= 30 else 53.5, "observed": True, "confidence": 0.9}
            for frame in range(1, 35)
        ]
        candidates = center_restart_candidates(rows, 65, 101, 2)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["status"], "needs_visual_confirmation")


class BallAdapterTests(unittest.TestCase):
    def test_merges_detector_and_tcn_without_changing_people(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            homographies = root / "img1"
            homographies.mkdir()
            for frame in range(1, 4):
                np.save(homographies / f"{frame:06d}.npy", np.eye(3))
            template = root / "template.png"
            cv2.imwrite(str(template), np.zeros((100, 100, 3), dtype=np.uint8))
            gsr_path = root / "SNGS-999.json"
            person = {"attributes": {"role": "player"}, "image_id": "3999000001", "id": "1"}
            legacy_ball = {"attributes": {"role": "ball"}, "image_id": "3999000001", "id": "2"}
            gsr_path.write_text(json.dumps({"predictions": [person, legacy_ball]}), encoding="utf-8")
            trajectory_path = root / "trajectory.jsonl"
            rows = [
                {"frame_index": 0, "final_center": [60, 70], "box": [57, 67, 63, 73],
                 "confidence": 0.8, "output_status": "detected"},
                {"frame_index": 1, "final_center": [61, 71], "box": None,
                 "confidence": None, "output_status": "completed_missing"},
                {"frame_index": 2, "final_center": None, "box": None,
                 "confidence": None, "output_status": "invalid"},
            ]
            trajectory_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            counts = merge_ball_trajectory(gsr_path, trajectory_path, homographies, template, "999", 3)
            merged = json.loads(gsr_path.read_text(encoding="utf-8"))["predictions"]
            self.assertEqual(counts, {"detected": 1, "completed_missing": 1, "invalid": 1,
                                      "removed_legacy_balls": 1})
            self.assertEqual(merged[0], person)
            self.assertEqual(len(merged), 3)
            self.assertEqual(merged[1]["bbox_pitch"]["x_bottom_middle"], 10.0)
            self.assertEqual(merged[2]["attributes"]["detection_status"], "completed_missing")


if __name__ == "__main__":
    unittest.main()
