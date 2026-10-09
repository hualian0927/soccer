import unittest

from tactical_analysis.projection_l1 import analyze_projection_l1


def synthetic_sequence(receiver_camera="A"):
    fps = 25
    ball = []
    for frame in range(1, 51):
        if frame <= 25:
            x = 10.0
        elif frame <= 35:
            x = 10.0 + (frame - 26) * 0.2
        else:
            x = 16.0 + (frame - 36) * 0.1
        ball.append({"frame": frame, "time_s": (frame - 1) / fps, "x_m": x,
                     "y_m": 20.0, "observed": True, "confidence": 0.9,
                     "source_camera": "A" if frame <= 35 else receiver_camera})
    people = [
        {"person_id": "A#1", "camera": "A", "color": "red", "role": "Player",
         "positions": [{"frame": row["frame"], "x_m": row["x_m"], "y_m": 20.0}
                       for row in ball if row["frame"] <= 35]},
        {"person_id": f"{receiver_camera}#2", "camera": receiver_camera,
         "color": "red", "role": "Player",
         "positions": [{"frame": row["frame"], "x_m": row["x_m"], "y_m": 20.0}
                       for row in ball if row["frame"] >= 36]},
    ]
    restart = [{"timestamp_sec": 1.0, "label_zh": "中圈开球候选",
                "evidence_start_sec": 0.5, "evidence_end_sec": 1.5,
                "evidence_zh": "静止后移动"}]
    return ball, people, restart


class ProjectionL1Tests(unittest.TestCase):
    def test_kickoff_gates_static_preplay_and_same_camera_transfer(self):
        ball, people, restarts = synthetic_sequence()
        result = analyze_projection_l1(ball, people, restarts, [], 2.0, 25)
        events = result["events"]
        self.assertEqual(result["summary"]["active_play_gate_sec"], 1.0)
        self.assertEqual(sum(event["event_type"] == "touch" for event in events), 2)
        self.assertEqual(sum(event["event_type"] == "pass" for event in events), 1)
        self.assertEqual(sum(event["event_type"] == "receive" for event in events), 1)
        self.assertTrue(all(event["timestamp_sec"] >= 1.0 for event in events
                            if event["event_type"] == "touch"))
        self.assertEqual(result["summary"]["confirmed_events"], 0)

    def test_cross_camera_transfer_is_not_a_pass(self):
        ball, people, restarts = synthetic_sequence(receiver_camera="B")
        result = analyze_projection_l1(ball, people, restarts, [], 2.0, 25)
        self.assertFalse(any(event["event_type"] in {"pass", "receive"}
                             for event in result["events"]))

    def test_unobserved_ball_cannot_assign_owner(self):
        ball, people, restarts = synthetic_sequence()
        for row in ball:
            row["observed"] = False
        result = analyze_projection_l1(ball, people, restarts, [], 2.0, 25)
        self.assertEqual([event["event_type"] for event in result["events"]], ["restart"])

    def test_same_player_reacquisition_is_not_a_pass(self):
        ball, people, restarts = synthetic_sequence()
        people[1]["person_id"] = people[0]["person_id"]
        result = analyze_projection_l1(ball, people, restarts, [], 2.0, 25)
        self.assertFalse(any(event["event_type"] in {"pass", "receive"}
                             for event in result["events"]))


if __name__ == "__main__":
    unittest.main()
