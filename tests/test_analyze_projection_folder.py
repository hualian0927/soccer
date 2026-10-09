import unittest

from analyze_projection_folder import sanitize_vision_result, vision_sample_times


class VisionResultTests(unittest.TestCase):
    def test_overlay_trajectory_is_not_pass_proof(self):
        event = {"event_type": "pass"}
        raw = {"decision": "visible", "observation_zh": "可见黄色传球轨迹从左侧球员传出"}
        result = sanitize_vision_result(event, raw)
        self.assertEqual(result["decision"], "uncertain")
        self.assertEqual(result["raw_decision"], "visible")
        self.assertEqual(raw["decision"], "visible")

    def test_clear_kicking_action_is_retained(self):
        event = {"event_type": "pass"}
        raw = {"decision": "visible", "observation_zh": "可见脚部触球后足球离开球员"}
        self.assertEqual(sanitize_vision_result(event, raw)["decision"], "visible")

    def test_pass_sampling_starts_before_estimated_release(self):
        event = {"event_id": "pass-1", "event_type": "pass", "timestamp_sec": 41.52,
                 "attributes": {"transfer_gap_sec": 1.84}}
        times = vision_sample_times(event, duration=60, fps=25)
        self.assertEqual(len(times), 16)
        self.assertLess(times[0], 41.52 - 1.84)
        self.assertGreater(times[-1], 41.52)
        self.assertEqual(len(set(times)), 16)

    def test_carry_sampling_includes_movement_endpoint(self):
        event = {"event_id": "carry-1", "event_type": "carry", "timestamp_sec": 51.76,
                 "attributes": {"duration_sec": 1.68}}
        times = vision_sample_times(event, duration=60, fps=25)
        self.assertLess(times[0], 51.76)
        self.assertGreater(times[-1], 51.76 + 1.68)


if __name__ == "__main__":
    unittest.main()
