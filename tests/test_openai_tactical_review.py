import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tactical_analysis.openai_review import (MODEL, BudgetExceeded, ReviewError, ResponsesReviewer,
    NoRedirect, read_response, responses_endpoint, parse_response, request_payload, validate_review)
from run_openai_tactical_review import audit_boundaries, compare_baseline, deduplicate, supplementation


def review(**changes):
    data = dict(decision="supported", title="传球", event_type="pass", subtype="pass", team="left",
        start_sec=10, end_sec=12, contact_time_sec=11, evidence_indices=[0, 2],
        observation="可见传递", interpretation="接应", advice="回看", limitation="仅抽样",
        restart_visible=False, ball_out_visible=False, keeper_identity_visible=False,
        incoming_shot_visible=False, contact_visible=True, scene_cut=False, needs_more_evidence=False,
        request=dict(kind="none", start_sec=10, end_sec=12, crop_xywh=[], reason=""))
    return {**data, **changes}


class OpenAIReviewTest(unittest.TestCase):
    frames = [{"time_sec": t, "jpeg": b"jpg", "frame": t * 30} for t in (10, 11, 12)]

    def test_valid_evidence(self):
        self.assertEqual(validate_review(review(), self.frames)["decision"], "supported")

    def test_outside_window_and_invalid_reference(self):
        for changes in ({"contact_time_sec": 14}, {"evidence_indices": [0, 8]}, {"start_sec": float("nan")},
                        {"start_sec": 12, "end_sec": 10}, {"decision": "yes"}):
            with self.assertRaises(ReviewError):
                validate_review(review(**changes), self.frames)

    def test_restart_requires_visible_restart(self):
        result = validate_review(review(event_type="set_piece", subtype="corner", ball_out_visible=True), self.frames)
        self.assertEqual(result["decision"], "uncertain")
        self.assertIn("重启", result["limitation"])

    def test_save_requires_incoming_shot(self):
        self.assertEqual(validate_review(review(event_type="goalkeeper", subtype="save", keeper_identity_visible=True), self.frames)["decision"], "uncertain")
        self.assertEqual(validate_review(review(event_type="goalkeeper", subtype="foot_pass", keeper_identity_visible=True), self.frames)["decision"], "supported")

    def test_keeper_pass_alias_and_imprecise_contact(self):
        result = validate_review(review(event_type="pass", subtype="foot_pass", keeper_identity_visible=True,
                                      contact_visible=False, contact_time_sec=None), self.frames)
        self.assertEqual(result["event_type"], "goalkeeper")
        self.assertEqual(result["decision"], "supported")
        result = validate_review(review(event_type="goalkeeper", subtype="save", keeper_identity_visible=True,
                                      incoming_shot_visible=True, contact_visible=False), self.frames)
        self.assertEqual(result["decision"], "uncertain")

    def test_single_frame_and_pending_evidence_cannot_publish(self):
        for changes in ({"evidence_indices": [0]}, {"needs_more_evidence": True}, {"event_type": "unknown"}):
            self.assertEqual(validate_review(review(**changes), self.frames)["decision"], "uncertain")

    def test_payload_has_images_schema_no_stored_response(self):
        payload = request_payload({"time": 11}, self.frames, {"left": "blue"})
        self.assertEqual(payload["model"], MODEL)
        self.assertFalse(payload["store"])
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertEqual(sum(p["type"] == "input_image" for p in payload["input"][0]["content"]), 3)
        previous = review()
        self.assertNotIn("previous_observation", json.dumps(request_payload({}, self.frames, {}, previous, True)))

    def test_refused_truncated_or_other_model_never_accepted(self):
        for response in ({"status": "incomplete"}, {"status": "completed", "model": "other"},
                         {"status": "completed", "model": MODEL, "output": [{"content": [{"type": "refusal"}]}]}):
            with self.assertRaises(ReviewError):
                parse_response(response)

    def test_completed_response(self):
        response = {"status": "completed", "model": MODEL, "output": [{"content": [{"type": "output_text", "text": json.dumps(review())}]}]}
        self.assertEqual(parse_response(response), review())

    def test_relay_model_identity_and_url_validation(self):
        self.assertEqual(responses_endpoint("https://codeapi.icu"), "https://codeapi.icu/v1/responses")
        self.assertEqual(responses_endpoint("https://codeapi.icu/v1/"), "https://codeapi.icu/v1/responses")
        for url in ("http://codeapi.icu", "https://user:key@codeapi.icu", "https://codeapi.icu?key=x"):
            with self.assertRaises(ReviewError):
                responses_endpoint(url)
        data = {"status": "completed", "model": "gpt-5.6-terra", "output": [{"content": [{"type": "output_text", "text": json.dumps(review())}]}]}
        self.assertEqual(parse_response(data, "gpt-5.6-terra"), review())
        for model in (MODEL, "gpt-5.6-terra-fallback"):
            with self.assertRaises(ReviewError):
                parse_response({**data, "model": model}, "gpt-5.6-terra")
        with self.assertRaises(ReviewError):
            NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example")

    def test_relay_no_monetary_cap_and_separate_ledger(self):
        import io
        response = {"status": "completed", "model": "gpt-5.6-terra", "id": "resp_relay",
            "usage": {"input_tokens": 1000000, "output_tokens": 5000},
            "output": [{"content": [{"type": "output_text", "text": json.dumps(review())}]}]}
        with tempfile.TemporaryDirectory() as tmp, patch("urllib.request.build_opener") as opener:
            stream = io.BytesIO(json.dumps(response).encode())
            stream.headers = {"Content-Type": "application/json"}
            opener.return_value.open.return_value = stream
            client = ResponsesReviewer("relay-secret", tmp, 1, None, model="gpt-5.6-terra", base_url="https://codeapi.icu")
            _, usage = client.call(request_payload({}, self.frames, {}))
            self.assertIsNone(usage["cost_upper_usd"])
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(json.loads(request.data)["model"], "gpt-5.6-terra")
            self.assertNotIn("relay-secret", (Path(tmp) / "api_usage.json").read_text())
            with self.assertRaises(BudgetExceeded):
                client.call(request_payload({}, self.frames, {}))
            with self.assertRaises(ReviewError):
                ResponsesReviewer("official-secret", tmp)

    def test_stream_requires_terminal_response(self):
        import io
        data = {"status": "completed", "model": "gpt-5.6-terra"}
        stream = io.BytesIO(b"event: response.created\ndata: {}\n\ndata: " +
                           json.dumps({"type": "response.completed", "response": data}).encode() + b"\n\n")
        stream.headers = {"Content-Type": "text/event-stream"}
        self.assertEqual(read_response(stream, True), data)
        stream = io.BytesIO(b"data: [DONE]\n")
        stream.headers = {"Content-Type": "text/event-stream"}
        with self.assertRaises(ReviewError):
            read_response(stream, True)

    def test_json_prompt_preserves_local_contract(self):
        import io
        response = {"status": "completed", "model": MODEL, "id": "resp_json",
                    "output": [{"content": [{"type": "output_text", "text": json.dumps(review())}]}]}
        with tempfile.TemporaryDirectory() as tmp, patch("urllib.request.build_opener") as opener:
            stream = io.BytesIO(json.dumps(response).encode())
            stream.headers = {"Content-Type": "application/json"}
            opener.return_value.open.return_value = stream
            client = ResponsesReviewer("secret", tmp, max_usd=None, model=MODEL,
                base_url="https://codeapi.icu/v1", response_format="json_prompt", reasoning_effort="low")
            payload = request_payload({}, self.frames, {})
            original = copy.deepcopy(payload)
            raw, usage = client.call(payload)
            self.assertEqual(payload, original)
            sent = json.loads(opener.return_value.open.call_args.args[0].data)
            self.assertNotIn("text", sent)
            self.assertIn("evidence_indices", sent["instructions"])
            self.assertEqual(validate_review(raw, self.frames)["decision"], "supported")
            self.assertEqual(usage["response_format"], "json_prompt")

    def test_budget_before_network_and_no_key_in_ledger(self):
        with tempfile.TemporaryDirectory() as tmp, patch("urllib.request.urlopen") as network:
            client = ResponsesReviewer("test-secret", Path(tmp), max_usd=.0001)
            with self.assertRaises(BudgetExceeded):
                client.call(request_payload({}, self.frames, {}))
            network.assert_not_called()
            self.assertFalse(client.ledger)

    def test_usage_record_and_cache_restart_budget(self):
        response = {"status": "completed", "model": MODEL, "id": "resp_test", "usage": {"input_tokens": 1000, "output_tokens": 500},
                    "output": [{"content": [{"type": "output_text", "text": json.dumps(review())}]}]}
        import io
        with tempfile.TemporaryDirectory() as tmp, patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            client = ResponsesReviewer("test-secret", Path(tmp), max_calls=1)
            result, usage = client.call(request_payload({}, self.frames, {}))
            self.assertEqual(result["decision"], "supported")
            ledger = (Path(tmp) / "api_usage.json").read_text()
            self.assertNotIn("test-secret", ledger)
            self.assertAlmostEqual(usage["cost_upper_usd"], .014)
            restarted = ResponsesReviewer("new-secret", Path(tmp), max_calls=1)
            with self.assertRaises(BudgetExceeded):
                restarted.call(request_payload({}, self.frames, {}))

    def test_extension_bounded_and_invalid_crop(self):
        task = {"time": 100}
        request = {"kind": "extend", "start_sec": 0, "end_sec": 999, "crop_xywh": []}
        self.assertEqual(supplementation(request, task, 300), (80, 120, None))
        with self.assertRaises(ReviewError):
            supplementation({**request, "kind": "crop", "crop_xywh": [.9, .9, .5, .5]}, task, 300)

    def test_duplicates_not_counted_and_baseline_not_truth(self):
        event = dict(id="a", time=10, subtype="pass", team="left", publishedForStatistics=True)
        events = [event, {**event, "id": "b", "time": 10.8}]
        original = copy.deepcopy(events)
        result = deduplicate(events)
        self.assertTrue(result[0]["publishedForStatistics"])
        self.assertFalse(result[1]["publishedForStatistics"])
        self.assertEqual(events, original)
        comparison = compare_baseline(result, {"events": [event]})
        self.assertIn("不是人工真值", comparison["note"])

    def test_waiting_outside_is_not_a_new_out_event(self):
        record = {"task": {"id": "out", "time": 100}, "signature": "first",
                  "review": review(event_type="ball_out", subtype="touchline", contact_time_sec=None)}
        audit = {**record, "signature": "second"}
        with patch("run_openai_tactical_review.review_task", return_value=audit) as tool:
            result = audit_boundaries([record], Path("video.mp4"), 300, Path("output"), None, {}, {})
        self.assertEqual(tool.call_args.args[0]["kind"], "boundary_transition")
        self.assertEqual(result[0]["review"]["decision"], "uncertain")
        self.assertEqual(result[0]["task"]["id"], "out")
        self.assertEqual(record["review"]["decision"], "supported")

    def test_boundary_prompt_requires_crossing_not_waiting(self):
        payload = request_payload({"kind": "boundary_transition"}, self.frames, {})
        self.assertIn("不能算新的出界", payload["instructions"])

    def test_same_boundary_event_deduplicates_despite_team_disagreement(self):
        first = dict(id="a", time=97, category="出界", subtype="unknown", team="left", publishedForStatistics=True)
        second = {**first, "id": "b", "time": 97.4, "subtype": "touchline", "team": "right"}
        result = deduplicate([first, second])
        self.assertIsNone(result[0]["team"])
        self.assertFalse(result[1]["publishedForStatistics"])
        self.assertEqual(first["team"], "left")

    def test_targeted_review_preferred_over_coarse_scan(self):
        first = dict(id="scan-00000", time=15, subtype="cross", team="left", publishedForStatistics=True)
        second = {**first, "id": "event-1", "time": 16.5}
        result = deduplicate([first, second])
        self.assertFalse(result[0]["publishedForStatistics"])
        self.assertTrue(result[1]["publishedForStatistics"])

    def test_explicit_boundary_rejection_is_preserved(self):
        record = {"task": {"id": "out", "time": 100}, "signature": "first", "review": review(event_type="ball_out")}
        audit = {**record, "review": review(event_type="ball_out", decision="rejected", contact_time_sec=None)}
        with patch("run_openai_tactical_review.review_task", return_value=audit):
            result = audit_boundaries([record], Path("video.mp4"), 300, Path("output"), None, {}, {})
        self.assertEqual(result[0]["review"]["decision"], "rejected")


if __name__ == "__main__":
    unittest.main()
