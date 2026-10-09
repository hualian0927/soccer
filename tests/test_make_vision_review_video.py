import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from make_vision_review_video import clock_time, load_reviews, review_display_time, review_style, wrap
from PIL import Image, ImageDraw
from make_vision_review_video import font


class VisionReviewVideoTest(unittest.TestCase):
    def test_clock_time(self):
        self.assertEqual(clock_time(299.9), "04:59")
        self.assertEqual(clock_time(300), "05:00")

    def test_review_style_keeps_uncertain_distinct(self):
        self.assertIn("证据不足", review_style({"fusion_status": "needs_human_review"})[0])
        self.assertIn("视觉支持", review_style({"fusion_status": "visual_supported_candidate"})[0])

    def test_wrap_long_chinese_text(self):
        draw = ImageDraw.Draw(Image.new("RGB", (400, 100)))
        lines = wrap(draw, "这是一个需要多帧证据复核的传球事件", 100, font(18))
        self.assertGreater(len(lines), 1)

    def test_set_piece_conclusion_waits_for_full_evidence(self):
        review = {
            "candidate": {"event_type": "set_piece_delivery_candidate", "time_sec": 54.133},
            "evidence_window_sec": [47.133, 78.133],
        }
        self.assertEqual(review_display_time(review), 78.133)

    def test_denser_review_replaces_older_same_event(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            video = root / "match.mp4"
            review = {
                "source_video": str(video), "status": "reviewed",
                "candidate": {"event_id": "corner-1", "event_type": "corner_candidate", "time_sec": 54.0},
                "vision_review": {"decision": "uncertain"},
            }
            (old / "corner.deepseek.json").write_text(json.dumps({**review, "frame_count": 8}))
            (new / "corner.deepseek.json").write_text(json.dumps({**review, "frame_count": 24}))
            loaded = load_reviews([old, new], video)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["frame_count"], 24)


if __name__ == "__main__":
    unittest.main()
