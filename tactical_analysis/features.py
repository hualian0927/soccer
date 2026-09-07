"""Reusable visual trajectory features for tactical analyzers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import AnalysisContext, FrameState, Observation


@dataclass(frozen=True)
class MotionFeature:
    vx: float
    vy: float
    speed: float
    acceleration: float
    direction_rad: float
    sample_gap_sec: float


MotionKey = tuple[int, str, int]


def build_motion_index(
    context: AnalysisContext,
    max_gap_sec: float = 0.6,
    smoothing: float = 0.55,
) -> dict[MotionKey, MotionFeature]:
    """Estimate pitch-plane motion with an exponential velocity smoother.

    Missing observations longer than ``max_gap_sec`` start a new motion segment,
    which prevents camera cuts and fragmented IDs from creating extreme speeds.
    """

    tracks: dict[tuple[str, int], list[Observation]] = {}
    for frame in context.frames:
        for observation in frame.observations:
            if not observation.has_pitch_position:
                continue
            tracks.setdefault((observation.role, observation.track_id), []).append(observation)

    result: dict[MotionKey, MotionFeature] = {}
    alpha = max(0.0, min(1.0, smoothing))
    for (role, track_id), observations in tracks.items():
        observations.sort(key=lambda item: item.frame)
        previous: Observation | None = None
        previous_vx = 0.0
        previous_vy = 0.0
        previous_speed = 0.0
        for observation in observations:
            if previous is None:
                previous = observation
                continue
            dt = observation.time_sec - previous.time_sec
            if dt <= 0 or dt > max_gap_sec:
                previous = observation
                previous_vx = previous_vy = previous_speed = 0.0
                continue
            raw_vx = (float(observation.pitch_x) - float(previous.pitch_x)) / dt
            raw_vy = (float(observation.pitch_y) - float(previous.pitch_y)) / dt
            vx = alpha * raw_vx + (1.0 - alpha) * previous_vx
            vy = alpha * raw_vy + (1.0 - alpha) * previous_vy
            speed = math.hypot(vx, vy)
            acceleration = (speed - previous_speed) / dt
            result[(observation.frame, role, track_id)] = MotionFeature(
                vx=round(vx, 4),
                vy=round(vy, 4),
                speed=round(speed, 4),
                acceleration=round(acceleration, 4),
                direction_rad=round(math.atan2(vy, vx), 5),
                sample_gap_sec=round(dt, 4),
            )
            previous = observation
            previous_vx, previous_vy, previous_speed = vx, vy, speed
    return result


def motion_for(
    motion_index: dict[MotionKey, MotionFeature],
    observation: Observation | None,
) -> MotionFeature | None:
    if observation is None:
        return None
    return motion_index.get((observation.frame, observation.role, observation.track_id))


def unique_team_players(frame: FrameState, team: str) -> list[Observation]:
    unique: dict[int, Observation] = {}
    for player in frame.players:
        if player.team == team and player.has_pitch_position:
            unique[player.track_id] = player
    return list(unique.values())


def frame_lookup(context: AnalysisContext) -> dict[int, FrameState]:
    return {frame.frame: frame for frame in context.frames}
