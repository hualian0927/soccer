"""Boundary-crossing candidates, without inferring a restart or last touch."""

import math

from ..base import TacticalAnalyzer, register_analyzer
from ..models import HALF_LENGTH, HALF_WIDTH, AnalyzerOutput, TimelineEvent


@register_analyzer("ball_out_of_play")
class BallOutOfPlayAnalyzer(TacticalAnalyzer):
    priority = "P0"
    description_zh = "连续轨迹越过球场边界后生成出界复核候选，不自动判定重启类型"

    def analyze(self, context):
        margin = float(self.config.get("outside_margin_m", 0.4))
        minimum_samples = int(self.config.get("minimum_outside_samples", 3))
        maximum_gap = float(self.config.get("maximum_gap_sec", 0.3))
        maximum_speed = float(self.config.get("maximum_speed_mps", 45))
        previous = None
        pending = []
        inside = None
        events = []
        last_event = -99.0
        for frame in context.frames:
            ball = frame.ball
            if ball is None or not ball.has_pitch_position:
                continue
            x, y = float(ball.pitch_x), float(ball.pitch_y)
            if not math.isfinite(x + y):
                previous, inside, pending = None, None, []
                continue
            continuous = False
            if previous:
                pf, pb = previous
                dt = frame.time_sec - pf.time_sec
                speed = math.hypot(x - pb.pitch_x, y - pb.pitch_y) / max(dt, 1e-6)
                continuous = 0 < dt <= maximum_gap and speed <= maximum_speed and pb.track_id == ball.track_id
            if not continuous:
                inside, pending = None, []
            if abs(x) <= HALF_LENGTH and abs(y) <= HALF_WIDTH:
                inside, pending = (frame, ball), []
            elif abs(x) > HALF_LENGTH + margin or abs(y) > HALF_WIDTH + margin:
                if inside and frame.time_sec - inside[0].time_sec <= 1.2:
                    pending.append((frame, ball))
                    if len(pending) >= minimum_samples and frame.time_sec - last_event >= 5:
                        first, first_ball = pending[0]
                        boundary = "touchline" if abs(first_ball.pitch_y) > HALF_WIDTH + margin else "goal_line"
                        label = "边线出界候选" if boundary == "touchline" else "球门线出界候选"
                        events.append(TimelineEvent(
                            event_id=f"ball-out-{len(events) + 1:03d}", event_type="ball_out_of_play_candidate",
                            frame=first.frame, time_sec=first.time_sec, team=None, actor_track_id=None,
                            target_track_id=None, confidence=0.6, label_zh=label, event_subtype="unknown",
                            start_x=inside[1].pitch_x, start_y=inside[1].pitch_y,
                            end_x=first_ball.pitch_x, end_y=first_ball.pitch_y,
                            evidence_start_sec=max(0, first.time_sec - 5),
                            evidence_end_sec=min(context.duration_sec, first.time_sec + 5),
                            metrics={"boundary": boundary, "outside_samples": len(pending),
                                     "restart_type": None, "requires_visual_boundary_review": True},
                        ))
                        last_event = frame.time_sec
                        inside, pending = None, []
            previous = (frame, ball)
        return AnalyzerOutput(
            analyzer=self.name, priority=self.priority, events=events,
            summary={"out_of_play_candidates": len(events)},
            warnings=["二维投影越界只产生候选；球整体出界、最后触球方和重启类型须分别核验。"],
        )
