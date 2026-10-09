import copy
import unittest

from build_assistant_review_version import build_review_bundle


class AssistantReviewVersionTest(unittest.TestCase):
    def setUp(self):
        self.event = {"id": "keeper", "category": "门将事件", "title": "扑救候选", "time": 10,
                      "evidenceFrames": [{"time": 9, "path": "a.jpg"}, {"time": 10, "path": "b.jpg"}]}
        self.source = {"source": {"duration_sec": 30}, "events": [self.event],
                       "layers": {"L2": {}, "L3": {}, "L4": {"items": []}}}
        self.annotations = {"reviewer": "Codex视觉助手", "scope": "抽样", "method": "direct", "source_video_sha256": "example",
                            "reviews": [{"id": "keeper", "category": "门将事件", "decision": "corrected", "subtype": "foot_pass",
                                         "time": 10, "title": "门将脚下出球", "observation": "站立踢球", "analysis": "非扑救",
                                         "limitation": "不计算成功率", "evidence_interval": [9, 11]}]}

    def test_preserves_original_and_never_claims_human_confirmation(self):
        before = copy.deepcopy(self.source)
        bundle = build_review_bundle(self.source, self.annotations)
        event = bundle["events"][0]
        self.assertEqual(self.source, before)
        self.assertEqual(event["originalCandidate"]["title"], "扑救候选")
        self.assertEqual(event["reviewStatus"], "candidate")
        self.assertFalse(bundle["reviewMetadata"]["human_ground_truth"])
        self.assertEqual(bundle["reviewMetadata"]["api_calls"], 0)
        self.assertIsNone(event["confidence"])

    def test_rejected_event_not_published_for_statistics(self):
        self.annotations["reviews"][0]["decision"] = "rejected"
        event = build_review_bundle(self.source, self.annotations)["events"][0]
        self.assertFalse(event["publishedForStatistics"])

    def test_duplicate_and_missing_reviews_are_rejected(self):
        self.annotations["reviews"] *= 2
        with self.assertRaises(ValueError):
            build_review_bundle(self.source, self.annotations)
        self.annotations["reviews"] = []
        with self.assertRaises(ValueError):
            build_review_bundle(self.source, self.annotations)

    def test_cannot_claim_unavailable_evidence(self):
        self.annotations["reviews"][0]["inspected_indices"] = [999]
        with self.assertRaises(ValueError):
            build_review_bundle(self.source, self.annotations)


if __name__ == "__main__":
    unittest.main()
