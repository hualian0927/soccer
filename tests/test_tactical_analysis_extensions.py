from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tactical_analysis.analyzers.events import EventTimelineAnalyzer
from tactical_analysis.analyzers.goalkeeper import GoalkeeperInterventionAnalyzer
from tactical_analysis.analyzers.defensive_gaps import DefensiveGapAnalyzer
from tactical_analysis.analyzers.numerical_superiority import NumericalSuperiorityAnalyzer
from tactical_analysis.analyzers.progression import extract_pass_records
from tactical_analysis.analyzers.set_pieces import SetPieceDeliveryAnalyzer, classify_set_piece
from tactical_analysis.analyzers.team_shape import _convex_hull_area, shape_sample
from tactical_analysis.models import AnalysisContext, AnalysisReport, AnalyzerOutput, FrameState, Observation, TimelineEvent
from tactical_analysis.reporting import build_highlight_manifest, build_match_summary


def player(frame: int, time_sec: float, track_id: int, x: float, y: float, team: str = "left") -> Observation:
    return Observation(frame, time_sec, track_id, "player", team, x, y)


def ball(frame: int, time_sec: float, x: float, y: float) -> Observation:
    return Observation(frame, time_sec, 99, "ball", None, x, y)


def goalkeeper(frame: int, time_sec: float, track_id: int, x: float, y: float, team: str = "right") -> Observation:
    return Observation(frame, time_sec, track_id, "goalkeeper", team, x, y)


class TacticalAnalysisExtensionsTest(unittest.TestCase):
    def test_touch_carry_pass_receive_chain(self) -> None:
        frames = [
            FrameState(1, 0.0, [player(1, 0.0, 1, 0.0, 0.0), player(1, 0.0, 2, 18.0, 0.0), ball(1, 0.0, 0.0, 0.0)]),
            FrameState(11, 1.0, [player(11, 1.0, 1, 8.0, 0.0), player(11, 1.0, 2, 18.0, 0.0), ball(11, 1.0, 8.0, 0.0)]),
            FrameState(21, 2.0, [player(21, 2.0, 1, 0.0, 0.0), player(21, 2.0, 2, 18.0, 0.0), ball(21, 2.0, 18.0, 0.0)]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            context = AnalysisContext(Path(directory) / "input.json", None, 10.0, frames)
            output = EventTimelineAnalyzer(
                {
                    "control_radius_m": 3.0,
                    "possession_confirm_samples": 1,
                    "possession_release_samples": 1,
                    "min_pass_distance_m": 4.0,
                    "min_carry_distance_m": 6.0,
                    "min_carry_progress_m": 4.0,
                    "min_carry_duration_sec": 0.5,
                }
            ).analyze(context)
        event_types = [event.event_type for event in output.events]
        self.assertIn("touch_candidate", event_types)
        self.assertIn("carry_candidate", event_types)
        self.assertIn("pass_candidate", event_types)
        self.assertIn("receive_candidate", event_types)

    def test_team_shape_lines_and_hull(self) -> None:
        observations = [
            player(1, 0.0, index + 1, x, y)
            for index, (x, y) in enumerate(
                [(-40, -20), (-38, 0), (-39, 20), (-12, -16), (-10, 0), (-11, 16), (18, -14), (20, 0), (19, 14)]
            )
        ]
        frame = FrameState(1, 0.0, observations)
        sample = shape_sample(frame, "left", "in_possession", 7)
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertGreater(sample["midfield_line_height_m"], sample["defensive_line_height_m"])
        self.assertGreater(sample["forward_line_height_m"], sample["midfield_line_height_m"])
        self.assertGreater(sample["convex_hull_area_m2"], 0)
        self.assertAlmostEqual(
            _convex_hull_area(np.array([[0, 0], [2, 0], [2, 2], [0, 2]])),
            4.0,
        )

    def test_product_summary_and_highlight_manifest(self) -> None:
        event = TimelineEvent(
            "event-0001", "shot_candidate", 101, 10.0, "left", 7, None, 0.9, "射门候选",
            {"in_penalty_area": True, "shot_channel": "center"},
        )
        report = AnalysisReport(
            "1.3.0", "now", {"duration_sec": 30.0}, {},
            [AnalyzerOutput("event_timeline", "P0", {"possession_proxy": {"left": 0.6}}, events=[event])],
            {}, [],
        )
        summary = build_match_summary(report)
        highlights = build_highlight_manifest(report)
        self.assertEqual(summary["event_counts"]["shot_candidate"], 1)
        self.assertEqual(summary["shots"][0]["shot_channel"], "center")
        self.assertEqual(len(highlights), 1)
        self.assertEqual(highlights[0]["type"], "shot_candidate")

    def test_shooting_structure_aggregates_distance_channel_and_pressure(self) -> None:
        shots = [
            TimelineEvent(
                "shot-1", "shot_candidate", 10, 1.0, "left", 7, None, 0.8, "射门候选",
                {
                    "shot_distance_m": 10.0,
                    "shot_angle_deg": 31.0,
                    "shot_channel": "center",
                    "in_penalty_area": True,
                    "defenders_within_5m": 0,
                    "xg_context_proxy": 0.31,
                },
            ),
            TimelineEvent(
                "shot-2", "shot_candidate", 20, 2.0, "left", 8, None, 0.7, "射门候选",
                {
                    "shot_distance_m": 27.0,
                    "shot_angle_deg": 9.0,
                    "shot_channel": "right",
                    "in_penalty_area": False,
                    "defenders_within_5m": 2,
                    "xg_context_proxy": 0.08,
                },
            ),
        ]
        structure = EventTimelineAnalyzer._shooting_structure(shots)["left"]
        self.assertEqual(structure["shot_candidates"], 2)
        self.assertEqual(structure["distance_buckets"], {"close": 1, "medium": 0, "long": 1})
        self.assertEqual(structure["channel_counts"], {"center": 1, "right": 1})
        self.assertEqual(structure["pressure_buckets"], {"low": 1, "medium": 0, "high": 1})

    def test_pass_records_have_single_priority_type(self) -> None:
        frames = [
            FrameState(1, 0.0, [player(1, 0.0, 1, 0.0, 0.0), player(1, 0.0, 2, 12.0, 0.0), ball(1, 0.0, 0.0, 0.0)]),
            FrameState(6, 0.5, [player(6, 0.5, 1, 1.0, 0.0), player(6, 0.5, 2, 12.0, 0.0), ball(6, 0.5, 1.0, 0.0)]),
            FrameState(11, 1.0, [player(11, 1.0, 1, 1.0, 0.0), player(11, 1.0, 2, 12.0, 0.0), ball(11, 1.0, 12.0, 0.0)]),
        ]
        context = AnalysisContext(Path("input.json"), None, 10.0, frames)
        records = extract_pass_records(
            context,
            {
                "control_radius_m": 3.0,
                "possession_confirm_samples": 1,
                "possession_release_samples": 1,
                "min_pass_distance_m": 4.0,
            },
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["pass_type"], "short_pass")

    def test_corner_delivery_reaches_first_controlled_landing(self) -> None:
        def corner_frame(frame: int, time_sec: float, ball_x: float, ball_y: float, receiver_x: float, receiver_y: float) -> FrameState:
            return FrameState(
                frame,
                time_sec,
                [
                    player(frame, time_sec, 1, 50.0, 32.0),
                    player(frame, time_sec, 2, receiver_x, receiver_y),
                    ball(frame, time_sec, ball_x, ball_y),
                ],
            )

        frames = [
            corner_frame(1, 0.0, 50.0, 32.0, 42.0, 12.0),
            corner_frame(5, 0.4, 50.0, 32.0, 42.0, 12.0),
            corner_frame(6, 0.5, 49.0, 30.0, 42.0, 12.0),
            corner_frame(9, 0.8, 42.0, 12.0, 42.0, 12.0),
            corner_frame(10, 0.9, 42.0, 12.0, 42.0, 12.0),
            corner_frame(11, 1.0, 42.0, 12.0, 42.0, 12.0),
        ]
        context = AnalysisContext(Path("input.json"), None, 10.0, frames)
        output = SetPieceDeliveryAnalyzer(
            {
                "control_radius_m": 4.0,
                "possession_confirm_samples": 1,
                "possession_release_samples": 1,
                "landing_control_radius_m": 4.0,
                "minimum_stationary_sec": 0.7,
                "candidate_cooldown_sec": 2.0,
            }
        ).analyze(context)
        self.assertEqual(len(output.events), 1)
        self.assertEqual(output.events[0].event_type, "set_piece_delivery_candidate")
        self.assertEqual(output.events[0].target_track_id, 2)
        self.assertTrue(output.events[0].metrics["retained_by_taking_team"])

    def test_set_piece_geometry_classifies_supported_restart_types(self) -> None:
        self.assertEqual(classify_set_piece(50.0, 32.0, "left", 1), "corner")
        self.assertEqual(classify_set_piece(0.5, -0.5, "left", 1), "kickoff")
        self.assertEqual(classify_set_piece(41.5, 0.0, "left", 1), "penalty")
        self.assertEqual(classify_set_piece(10.0, 33.0, "left", 1), "throw_in")
        self.assertEqual(classify_set_piece(-48.0, 2.0, "left", 1), "goal_kick")
        self.assertEqual(classify_set_piece(20.0, 8.0, "left", 1), "unknown")

    def test_goalkeeper_event_requires_sustained_ball_proximity(self) -> None:
        frames = [
            FrameState(1, 0.0, [goalkeeper(1, 0.0, 10, 49.0, 0.0), ball(1, 0.0, 47.0, 0.0)]),
            FrameState(2, 0.1, [goalkeeper(2, 0.1, 10, 49.0, 0.0), ball(2, 0.1, 47.5, 0.0)]),
        ]
        for index, distance in enumerate((12.0, 10.0, 8.0, 5.0, 2.0, 0.5), 11):
            time_sec = 1.0 + (index - 11) * 0.12
            frames.append(
                FrameState(index, time_sec, [goalkeeper(index, time_sec, 20, 49.0, 0.0), ball(index, time_sec, 49.0 - distance, 0.0)])
            )
        context = AnalysisContext(Path("input.json"), None, 10.0, frames)
        output = GoalkeeperInterventionAnalyzer(
            {
                "control_radius_m": 3.0,
                "possession_confirm_samples": 1,
                "possession_release_samples": 1,
                "intervention_radius_m": 5.5,
                "approach_radius_m": 14.0,
                "minimum_proximity_duration_sec": 0.4,
                "minimum_proximity_samples": 3,
                "minimum_approach_distance_m": 4.5,
            }
        ).analyze(context)
        self.assertEqual(len(output.events), 1)
        self.assertAlmostEqual(output.events[0].time_sec, 1.6)
        self.assertLess(output.events[0].metrics["minimum_ball_distance_m"], 1.0)
        self.assertGreater(output.events[0].metrics["approach_distance_drop_m"], 10.0)

    def test_dynamic_numerical_superiority_requires_sustained_advantage(self) -> None:
        frames = []
        for index, time_sec in enumerate((0.0, 0.2, 0.4, 0.6), 1):
            observations = [
                player(index, time_sec, 1, 0.0, 0.0),
                player(index, time_sec, 2, 3.0, 2.0),
                player(index, time_sec, 3, 3.0, -2.0),
                player(index, time_sec, 101, 8.0, 0.0, "right"),
                player(index, time_sec, 102, 30.0, 20.0, "right"),
                player(index, time_sec, 103, 30.0, -20.0, "right"),
                ball(index, time_sec, 0.0, 0.0),
            ]
            frames.append(FrameState(index, time_sec, observations))
        context = AnalysisContext(Path("input.json"), None, 10.0, frames)
        output = NumericalSuperiorityAnalyzer(
            {
                "control_radius_m": 3.0,
                "possession_confirm_samples": 1,
                "possession_release_samples": 1,
                "sample_step_frames": 1,
                "local_radius_m": 12.0,
                "minimum_advantage": 2,
                "minimum_attackers": 3,
                "minimum_episode_sec": 0.3,
            }
        ).analyze(context)
        self.assertEqual(len(output.events), 1)
        self.assertEqual(output.events[0].event_type, "numerical_superiority_candidate")
        self.assertGreaterEqual(output.events[0].metrics["peak_advantage"], 2)

    def test_defensive_gap_combines_line_spacing_and_low_pressure(self) -> None:
        frames = []
        defender_points = [(-10, -12), (-10, 12), (5, -12), (5, 12), (25, -14), (25, 0), (25, 14)]
        for index, time_sec in enumerate((0.0, 0.2, 0.4, 0.6), 1):
            observations = [
                player(index, time_sec, 1, 15.0, 0.0),
                player(index, time_sec, 2, 14.0, 3.0),
                ball(index, time_sec, 15.0, 0.0),
            ]
            observations.extend(
                player(index, time_sec, 100 + offset, x, y, "right")
                for offset, (x, y) in enumerate(defender_points)
            )
            frames.append(FrameState(index, time_sec, observations))
        context = AnalysisContext(Path("input.json"), None, 10.0, frames)
        output = DefensiveGapAnalyzer(
            {
                "control_radius_m": 3.0,
                "possession_confirm_samples": 1,
                "possession_release_samples": 1,
                "sample_step_frames": 1,
                "minimum_visible_defenders": 7,
                "minimum_line_gap_m": 12.0,
                "pressure_radius_m": 8.0,
                "minimum_episode_sec": 0.3,
            }
        ).analyze(context)
        self.assertEqual(len(output.events), 1)
        self.assertEqual(output.events[0].event_type, "defensive_gap_candidate")
        self.assertGreaterEqual(output.events[0].metrics["maximum_line_gap_m"], 12.0)


if __name__ == "__main__":
    unittest.main()
