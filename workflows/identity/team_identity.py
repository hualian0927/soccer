"""Track-level team identity refinement from user-provided kit colors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrackIdentity:
    track_id: int
    identity: str
    confidence: float
    margin: float
    evidence_frames: int
    scores: dict[str, float]
    method: str = "color_vote"


def _row_weight(row: pd.Series) -> float:
    score = float(row.get("score", 0.5) or 0.5)
    width = max(1.0, float(row.get("w", 1.0) or 1.0))
    height = max(1.0, float(row.get("h", 1.0) or 1.0))
    # Larger crops carry more color information, but capping prevents close-ups dominating a track.
    size_quality = min(2.5, np.sqrt(width * height) / 35.0)
    return max(0.05, min(1.0, score)) * max(0.25, size_quality)


def infer_track_identities(
    track_df: pd.DataFrame,
    color_groups: dict[str, set[str]],
    minimum_evidence_frames: int = 3,
    minimum_confidence: float = 0.58,
    minimum_margin: float = 0.18,
) -> dict[int, TrackIdentity]:
    """Aggregate noisy frame colors into one stable identity per track."""
    color_to_identities: dict[str, set[str]] = defaultdict(set)
    for identity, colors in color_groups.items():
        for color in colors:
            color_to_identities[str(color).strip().lower()].add(identity)

    decisions: dict[int, TrackIdentity] = {}
    if track_df.empty:
        return decisions
    for raw_track_id, rows in track_df.groupby("track_id", sort=False):
        scores = defaultdict(float)
        evidence_frames = 0
        for _, row in rows.iterrows():
            color = str(row.get("color", "unknown")).strip().lower()
            identities = color_to_identities.get(color, set())
            # Overlapping hints are ambiguous and must not vote for both teams.
            if len(identities) != 1:
                continue
            identity = next(iter(identities))
            scores[identity] += _row_weight(row)
            evidence_frames += 1
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        identity, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(scores.values())
        confidence = best / max(total, 1e-6)
        margin = (best - second) / max(total, 1e-6)
        enough_frames = evidence_frames >= minimum_evidence_frames or confidence >= 0.78
        if enough_frames and confidence >= minimum_confidence and margin >= minimum_margin:
            track_id = int(raw_track_id)
            decisions[track_id] = TrackIdentity(
                track_id=track_id,
                identity=identity,
                confidence=round(confidence, 4),
                margin=round(margin, 4),
                evidence_frames=evidence_frames,
                scores={name: round(value, 3) for name, value in scores.items()},
            )
    return decisions


def _identity_by_color(color_groups: dict[str, set[str]]) -> dict[str, str]:
    identities: dict[str, set[str]] = defaultdict(set)
    for identity, colors in color_groups.items():
        for color in colors:
            identities[str(color).strip().lower()].add(identity)
    return {
        color: next(iter(candidates))
        for color, candidates in identities.items()
        if len(candidates) == 1
    }


def _smooth_identity_labels(
    labels: list[str | None],
    window: int,
    minimum_votes: int,
    minimum_ratio: float,
) -> list[str | None]:
    radius = max(1, window // 2)
    smoothed: list[str | None] = []
    for index in range(len(labels)):
        nearby = [
            label
            for label in labels[max(0, index - radius): min(len(labels), index + radius + 1)]
            if label is not None
        ]
        if not nearby:
            smoothed.append(None)
            continue
        counts = defaultdict(int)
        for label in nearby:
            counts[label] += 1
        identity, votes = max(counts.items(), key=lambda item: item[1])
        smoothed.append(identity if votes >= minimum_votes and votes / len(nearby) >= minimum_ratio else None)
    series = pd.Series(smoothed, dtype="object").ffill().bfill()
    return [None if pd.isna(value) else str(value) for value in series.tolist()]


def _merge_short_runs(
    runs: list[tuple[int, int, str | None]],
    minimum_run: int,
) -> list[tuple[int, int, str | None]]:
    merged = list(runs)
    while len(merged) > 1:
        short_index = next(
            (index for index, (start, end, _) in enumerate(merged) if end - start < minimum_run),
            None,
        )
        if short_index is None:
            break
        start, end, _ = merged[short_index]
        if short_index == 0:
            next_start, next_end, next_label = merged[1]
            merged[0:2] = [(start, next_end, next_label)]
        elif short_index == len(merged) - 1:
            prev_start, _, prev_label = merged[-2]
            merged[-2:] = [(prev_start, end, prev_label)]
        else:
            prev_start, prev_end, prev_label = merged[short_index - 1]
            next_start, next_end, next_label = merged[short_index + 1]
            if prev_end - prev_start >= next_end - next_start:
                merged[short_index - 1: short_index + 1] = [(prev_start, end, prev_label)]
            else:
                merged[short_index: short_index + 2] = [(start, next_end, next_label)]
    return merged


def split_identity_inconsistent_tracks(
    track_df: pd.DataFrame,
    color_groups: dict[str, set[str]],
    maximum_gap_frames: int = 15,
    smoothing_window: int = 13,
    minimum_switch_run: int = 10,
    minimum_votes: int = 3,
    minimum_ratio: float = 0.62,
) -> tuple[pd.DataFrame, list[dict]]:
    """Split IDs reused across cuts or sustained kit-appearance changes.

    Long broadcast clips frequently reuse or incorrectly merge one tracker ID for
    athletes from different teams. A global majority vote then propagates one bad
    team label through the whole clip. This creates shorter, identity-consistent
    tracklets before team voting, following the split-before-aggregate strategy
    used by recent SoccerNet GSR systems.
    """

    if track_df.empty:
        return track_df.copy(), []
    result = track_df.copy()
    color_identity = _identity_by_color(color_groups)
    next_track_id = int(pd.to_numeric(result["track_id"], errors="coerce").max()) + 1
    records: list[dict] = []

    role_lower = result["role"].astype(str).str.lower()
    people = result[role_lower != "ball"]
    for raw_track_id, rows in people.groupby("track_id", sort=False):
        rows = rows.sort_values("frame")
        frames = pd.to_numeric(rows["frame"], errors="coerce").fillna(0).astype(int).to_numpy()
        if not len(frames):
            continue
        chunk_starts = [0]
        chunk_starts.extend((np.flatnonzero(np.diff(frames) > maximum_gap_frames) + 1).tolist())
        chunk_starts.append(len(rows))
        source_segments: list[tuple[np.ndarray, str | None]] = []
        for chunk_start, chunk_end in zip(chunk_starts, chunk_starts[1:]):
            chunk = rows.iloc[chunk_start:chunk_end]
            raw_labels = [
                color_identity.get(str(color).strip().lower())
                for color in chunk["color"].tolist()
            ]
            labels = _smooth_identity_labels(
                raw_labels,
                window=smoothing_window,
                minimum_votes=minimum_votes,
                minimum_ratio=minimum_ratio,
            )
            runs: list[tuple[int, int, str | None]] = []
            run_start = 0
            for index in range(1, len(labels)):
                if labels[index] != labels[run_start]:
                    runs.append((run_start, index, labels[run_start]))
                    run_start = index
            runs.append((run_start, len(labels), labels[run_start] if labels else None))
            for start, end, label in _merge_short_runs(runs, minimum_switch_run):
                source_segments.append((chunk.index.to_numpy()[start:end], label))

        if len(source_segments) <= 1:
            continue
        generated_ids = []
        for segment_index, (indices, label) in enumerate(source_segments):
            assigned_id = int(raw_track_id) if segment_index == 0 else next_track_id
            if segment_index:
                next_track_id += 1
            result.loc[indices, "track_id"] = assigned_id
            generated_ids.append(assigned_id)
            records.append(
                {
                    "source_track_id": int(raw_track_id),
                    "track_id": assigned_id,
                    "start_frame": int(result.loc[indices, "frame"].min()),
                    "end_frame": int(result.loc[indices, "frame"].max()),
                    "rows": int(len(indices)),
                    "smoothed_identity": label,
                }
            )
        if len(set(generated_ids)) != len(generated_ids):
            raise RuntimeError(f"Duplicate generated track IDs for source track {raw_track_id}")

    result["track_id"] = pd.to_numeric(result["track_id"], errors="raise").astype(int)
    return result, records


def apply_track_identities(
    track_df: pd.DataFrame,
    decisions: dict[int, TrackIdentity],
    max_referees_per_frame: int = 3,
) -> pd.DataFrame:
    """Apply stable decisions while retaining unknown tracks for later fallback logic."""
    if track_df.empty or not decisions:
        return track_df
    result = track_df.copy()
    role_lower = result["role"].astype(str).str.lower()
    non_ball = role_lower != "ball"
    result.loc[non_ball, "team"] = -1
    result.loc[non_ball & (role_lower == "referee"), "role"] = "Player"

    identity_by_track = {
        int(track_id): decision.identity
        for track_id, decision in decisions.items()
    }
    row_identity = result["track_id"].map(identity_by_track)
    role_by_identity = {
        "team0": "Player",
        "team1": "Player",
        "goalkeeper_team0": "Goalkeeper",
        "goalkeeper_team1": "Goalkeeper",
        "referee": "Referee",
    }
    team_by_identity = {
        "team0": 0,
        "team1": 1,
        "goalkeeper_team0": 0,
        "goalkeeper_team1": 1,
        "referee": -1,
    }
    assigned = non_ball & row_identity.notna()
    result.loc[assigned, "role"] = row_identity[assigned].map(role_by_identity)
    result.loc[assigned, "team"] = row_identity[assigned].map(team_by_identity).astype(int)

    # A color error cannot create more than the configured number of officials in one frame.
    confidence_by_track = {track_id: decision.confidence for track_id, decision in decisions.items()}
    for _, frame_rows in result.groupby("frame", sort=False):
        referee_rows = frame_rows[frame_rows["role"].astype(str).str.lower() == "referee"]
        if len(referee_rows) <= max_referees_per_frame:
            continue
        keep = sorted(
            referee_rows.index,
            key=lambda index: (
                confidence_by_track.get(int(result.at[index, "track_id"]), 0.0),
                float(result.at[index, "h"]),
            ),
            reverse=True,
        )[:max_referees_per_frame]
        demote = referee_rows.index.difference(keep)
        result.loc[demote, "role"] = "Player"
        result.loc[demote, "team"] = -1
    return result


def identity_report_rows(decisions: Iterable[TrackIdentity]) -> list[dict]:
    return [
        {
            "track_id": decision.track_id,
            "identity": decision.identity,
            "confidence": decision.confidence,
            "margin": decision.margin,
            "evidence_frames": decision.evidence_frames,
            "scores": decision.scores,
            "method": decision.method,
        }
        for decision in sorted(decisions, key=lambda item: item.track_id)
    ]
