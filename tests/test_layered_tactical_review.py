import unittest
from collections import deque
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from workflows.review.build_layered_tactical_review import build_bundle
from tactical_analysis.analyzers.ball_out import BallOutOfPlayAnalyzer
from tactical_analysis.analyzers.goalkeeper import image_ball_distance
from tactical_analysis.models import AnalysisContext, FrameState, Observation
from workflows.visualization.make_tactical_visualization_video import Detection, draw_frame


class LayeredReviewTest(unittest.TestCase):
    def test_boxes_only_preserves_pixels_outside_box(self):
        frame = np.full((180, 320, 3), 80, dtype=np.uint8)
        state = Mock()
        detection = Detection(1, 1, "player", "left", {"x": 100, "y": 50, "w": 30, "h": 70}, 0, 0)
        output = draw_frame(frame, 1, 30, [detection], deque(), state, boxes_only=True)
        mask = np.ones(frame.shape[:2], dtype=bool)
        mask[46:125, 96:135] = False
        self.assertTrue(np.array_equal(output[mask], frame[mask]))
        self.assertFalse(np.array_equal(output, frame))
        state.update.assert_not_called()

    def context(self, points):
        return AnalysisContext(Path("input.json"), None, 10, [
            FrameState(i + 1, i / 10, [Observation(i + 1, i / 10, 1, "ball", None, x, y)])
            for i, (x, y) in enumerate(points)
        ])

    def test_out_of_play_requires_continuous_crossing(self):
        analyzer = BallOutOfPlayAnalyzer({})
        valid = analyzer.analyze(self.context([(0, 33.5), (0, 34.5), (0, 34.7), (0, 34.9)]))
        self.assertEqual(len(valid.events), 1)
        self.assertEqual(valid.events[0].event_subtype, "unknown")
        self.assertIsNone(valid.events[0].metrics["restart_type"])
        jump = analyzer.analyze(self.context([(0, 0), (0, 36), (0, 37), (0, 38)]))
        self.assertEqual(jump.events, [])

    def test_image_distance_rejects_pitch_only_coincidence(self):
        keeper = Observation(1, 0, 1, "goalkeeper", "left", 50, 0, {"x": 100, "y": 100, "w": 40, "h": 80})
        ball = Observation(1, 0, 2, "ball", None, 50, 0, {"x": 800, "y": 100, "w": 8, "h": 8})
        self.assertGreater(image_ball_distance(keeper, ball), 0.65)

    def test_rejected_shot_cannot_create_attacking_chain(self):
        shot = {"event_id": "shot-1", "event_type": "shot_candidate", "time_sec": 10}
        chain = {"event_id": "chain-1", "event_type": "attacking_chain_candidate", "time_sec": 10, "metrics": {"action_count": 3, "pass_count": 2}}
        report = {"source": {"duration_sec": 30}, "analyzer_outputs": [
            {"analyzer": "event_timeline", "events": [shot]},
            {"analyzer": "attacking_chains", "events": [chain]},
        ]}
        reviews = {"shot-1": {"status": "reviewed", "fusion_status": "rejected_candidate",
                              "vision_review": {"confidence": 0.8, "short_reason_zh": "仅为传球"}}}
        bundle = build_bundle(report, [shot], reviews)
        self.assertEqual(bundle["layers"]["L3"]["items"], [])
        shooting = bundle["layers"]["L2"]["counts"][0]
        self.assertEqual(shooting["supported"], 0)
        self.assertEqual(shooting["rejected"], 1)

    def test_conflicting_keeper_narrative_is_not_published_as_fact(self):
        event = {"event_id": "gk-1", "event_type": "goalkeeper_intervention_candidate", "time_sec": 208}
        report = {"source": {"duration_sec": 300}, "analyzer_outputs": []}
        review = {"status": "reviewed", "fusion_status": "conflict_needs_human_review",
                  "vision_review": {"decision": "uncertain", "confidence": 0.9,
                                    "observation_zh": "倒地扑救", "goalkeeper": {"action_type": "save"}}}
        result = build_bundle(report, [event], {"gk-1": review})["events"][0]
        self.assertEqual(result["title"], "门将动作待复核")
        self.assertNotIn("倒地扑救", result["observation"])
        self.assertEqual(result["modelOpinions"][0]["observation"], "倒地扑救")


if __name__ == "__main__":
    unittest.main()
