import unittest

from tactical_analysis.l1 import _event_outcome, _evidence_window, _summary_text
from tactical_analysis.models import TimelineEvent
from review_tactical_candidates_with_vision import (
    build_request,
    evidence_bounds,
    find_candidate,
    fusion_status,
    sample_times,
    validate_review,
)


EVENT = {
    "event_id": "event-1",
    "event_type": "shot_candidate",
    "time_sec": 7.0,
    "confidence": 0.7,
    "metrics": {"shot_distance_m": 20.0, "private_internal": "omit"},
}


class VisionCandidateReviewTest(unittest.TestCase):
    def test_finds_supported_event(self):
        report = {"analyzer_outputs": [{"events": [EVENT]}]}
        self.assertEqual(find_candidate(report, "event-1"), EVENT)
        with self.assertRaises(ValueError):
            find_candidate(report, "missing")

    def test_sample_times_clamp_to_video(self):
        samples = sample_times({**EVENT, "time_sec": 0.3}, 3.0, 4)
        self.assertEqual(samples[0], 0.0)
        self.assertLess(samples[-1], 3.0)
        dense = sample_times(EVENT, 20.0, 5, 1.0, 1.0)
        self.assertEqual(dense, [6.0, 6.5, 7.0, 7.5, 8.0])

    def test_review_uses_requested_five_second_window(self):
        corner = {
            "event_type": "corner_candidate",
            "time_sec": 54.133,
            "evidence_start_sec": 50.133,
            "evidence_end_sec": 62.133,
        }
        self.assertEqual(evidence_bounds(corner, 300, 5, 5), (49.133, 59.133))
        samples = sample_times(corner, 300, 18)
        self.assertEqual((samples[0], samples[-1]), (49.133, 59.133))
        with self.assertRaises(ValueError):
            sample_times({**EVENT, "time_sec": 301}, 300, 21)
        timeline_event = TimelineEvent(
            event_id="corner-1", event_type="corner_candidate", frame=1625,
            time_sec=54.133, team="right", actor_track_id=None, target_track_id=None,
            confidence=0.72, label_zh="角球候选", event_subtype="corner",
            evidence_start_sec=50.133, evidence_end_sec=62.133,
        )
        self.assertEqual(_evidence_window(timeline_event, 300), (47.133, 78.133))

    def test_ball_out_does_not_confirm_restart(self):
        candidate = {**EVENT, "event_type": "set_piece_delivery_candidate", "event_subtype": "corner"}
        review = {"decision": "confirmed", "event_type": "corner", "event_subtype": "corner",
                  "ball_out_of_play": {"observed": True}, "restart_observed": False}
        self.assertEqual(fusion_status(candidate, review), "out_of_play_supported_restart_unconfirmed")

    def test_keeper_save_requires_identity_contact_and_incoming_shot(self):
        candidate = {**EVENT, "event_type": "goalkeeper_intervention_candidate"}
        review = {"decision": "confirmed", "event_type": "goalkeeper_intervention",
                  "goalkeeper": {"identity_visible": True, "ball_contact_visible": True,
                                 "incoming_shot_visible": False, "action_type": "save"}}
        self.assertEqual(fusion_status(candidate, review), "needs_human_review")
        review["goalkeeper"]["action_type"] = "foot_pass"
        self.assertEqual(fusion_status(candidate, review), "visual_supported_candidate")

    def test_out_of_bounds_landing_is_not_reported_as_control(self):
        event = TimelineEvent(
            event_id="corner-1", event_type="set_piece_delivery_candidate", frame=1625,
            time_sec=54.133, team="right", actor_track_id=None, target_track_id=None,
            confidence=0.72, label_zh="角球落点候选", event_subtype="corner",
            outcome="opponent_control", metrics={"landing_zone": "out_of_bounds", "landing_team": "left"},
        )
        self.assertEqual(_event_outcome(event), "unknown")
        self.assertIn("待原片复核", _summary_text(event, "corner", _event_outcome(event)))

    def test_request_uses_images_in_user_message(self):
        payload = build_request(EVENT, [{"time_sec": 7.0, "jpeg": b"jpeg"}])
        self.assertEqual(payload["model"], "deepseek-flash")
        content = payload["messages"][0]["content"]
        self.assertEqual(content[-1]["type"], "image_url")
        self.assertNotIn("private_internal", content[0]["text"])

    def test_restart_request_hides_geometry_subtype(self):
        corner = {
            **EVENT, "event_type": "set_piece_delivery_candidate", "event_subtype": "corner",
            "metrics": {"set_piece_type": "corner", "landing_sec": 9.0},
        }
        payload = build_request(corner, [{"time_sec": 7.0, "jpeg": b"jpeg"}])
        prompt = payload["messages"][0]["content"][0]["text"]
        self.assertIn('"candidate_type":"possible_restart"', prompt)
        self.assertNotIn('"set_piece_type":"corner"', prompt)

    def test_review_validation_and_fusion(self):
        review = {
            "decision": "confirmed",
            "event_type": "shot",
            "confidence": 0.7,
            "evidence_frame_indices": [0],
            "short_reason_zh": "可见射门动作",
        }
        self.assertEqual(validate_review(review, 2), review)
        self.assertEqual(fusion_status(EVENT, review), "visual_supported_candidate")
        self.assertEqual(
            fusion_status(EVENT, {**review, "event_type": "pass"}),
            "conflict_needs_human_review",
        )
        self.assertEqual(
            fusion_status(EVENT, {**review, "decision": "uncertain"}),
            "needs_human_review",
        )
        self.assertEqual(
            fusion_status({**EVENT, "event_type": "set_piece_delivery_candidate"}, {**review, "event_type": "corner"}),
            "visual_supported_candidate",
        )
        corner = {**EVENT, "event_type": "set_piece_delivery_candidate", "event_subtype": "corner"}
        self.assertEqual(
            fusion_status(corner, {**review, "event_type": "set_piece_delivery", "event_subtype": "throw_in"}),
            "conflict_needs_human_review",
        )
        self.assertEqual(
            fusion_status(corner, {**review, "event_type": "set_piece_delivery", "event_subtype": "corner_kick"}),
            "visual_supported_candidate",
        )
        with self.assertRaises(ValueError):
            validate_review({**review, "evidence_frame_indices": [2]}, 2)


if __name__ == "__main__":
    unittest.main()
