from __future__ import annotations

import unittest

import cv2
import numpy as np
import pandas as pd

from workflows.identity.jersey_color import estimate_jersey_color
from workflows.identity.team_identity import apply_track_identities, infer_track_identities, split_identity_inconsistent_tracks


class TeamIdentityTest(unittest.TestCase):
    def test_torso_color_ignores_green_crop_border(self) -> None:
        crop = np.full((100, 50, 3), (45, 145, 45), dtype=np.uint8)
        cv2.rectangle(crop, (13, 8), (37, 55), (210, 95, 35), -1)
        estimate = estimate_jersey_color(crop)
        self.assertEqual(estimate.label, "blue")

    def test_track_vote_locks_team_despite_frame_noise(self) -> None:
        rows = []
        for frame, color in enumerate(["blue"] * 8 + ["white"] * 2, 1):
            rows.append([frame, 10, 10, 10, 30, 70, 0.9, "Player", color, -1])
        for frame, color in enumerate(["white"] * 7 + ["blue"], 1):
            rows.append([frame, 20, 10, 10, 28, 68, 0.85, "Player", color, -1])
        for frame, color in enumerate(["red"] * 5, 1):
            rows.append([frame, 30, 10, 10, 25, 65, 0.8, "Player", color, -1])
        frame = pd.DataFrame(
            rows,
            columns=["frame", "track_id", "x", "y", "w", "h", "score", "role", "color", "team"],
        )
        decisions = infer_track_identities(
            frame,
            {"team0": {"blue"}, "team1": {"white"}, "referee": {"red"}},
        )
        self.assertEqual(decisions[10].identity, "team0")
        self.assertEqual(decisions[20].identity, "team1")
        self.assertEqual(decisions[30].identity, "referee")
        refined = apply_track_identities(frame, decisions)
        self.assertEqual(set(refined.loc[refined.track_id == 10, "team"]), {0})
        self.assertEqual(set(refined.loc[refined.track_id == 20, "team"]), {1})
        self.assertEqual(set(refined.loc[refined.track_id == 30, "role"]), {"Referee"})

    def test_mixed_team_track_is_split_before_identity_vote(self) -> None:
        rows = []
        for frame, color in enumerate(["white"] * 20 + ["blue"] * 22, 1):
            rows.append([frame, 7, 10, 10, 28, 68, 0.9, "Player", color, -1])
        frame = pd.DataFrame(
            rows,
            columns=["frame", "track_id", "x", "y", "w", "h", "score", "role", "color", "team"],
        )
        split, records = split_identity_inconsistent_tracks(
            frame,
            {"team0": {"blue"}, "team1": {"white"}},
            maximum_gap_frames=15,
            smoothing_window=9,
            minimum_switch_run=6,
        )
        self.assertEqual(split.track_id.nunique(), 2)
        self.assertEqual(len(records), 2)
        decisions = infer_track_identities(split, {"team0": {"blue"}, "team1": {"white"}})
        self.assertEqual({decision.identity for decision in decisions.values()}, {"team0", "team1"})


if __name__ == "__main__":
    unittest.main()
