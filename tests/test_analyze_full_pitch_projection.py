import unittest

import pandas as pd

from analyze_full_pitch_projection import deduplicate_players, sanitize_vision, shape_metrics


class FullPitchProjectionTest(unittest.TestCase):
    def test_cross_camera_same_color_is_deduplicated(self):
        rows = pd.DataFrame([
            {"camera": "left", "x_m": 10.0, "y_m": 50.0},
            {"camera": "right", "x_m": 10.5, "y_m": 50.2},
            {"camera": "right", "x_m": 25.0, "y_m": 70.0},
        ])
        points, merged = deduplicate_players(rows)
        self.assertEqual((len(points), merged), (2, 1))

    def test_shape_needs_enough_visible_players(self):
        import numpy as np

        self.assertIsNone(shape_metrics(np.array([[1, 1], [2, 2]]), 65, 101))

    def test_model_action_claim_is_withheld(self):
        result = sanitize_vision({"summary_zh": "红队发起角球", "observations_zh": []})
        self.assertEqual(result["status"], "withheld_unsupported_action_claim")

    def test_unverified_attacking_direction_is_withheld(self):
        result = sanitize_vision({"summary_zh": "红队重心更靠前", "observations_zh": []})
        self.assertEqual(result["status"], "withheld_unsupported_action_claim")


if __name__ == "__main__":
    unittest.main()
