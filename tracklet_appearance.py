"""CLIP tracklet aggregation for stable football team affiliation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from jersey_model.CLIPFinetune import CLIPFinetune
from team_identity import TrackIdentity


def _sample_rows(rows: pd.DataFrame, maximum_samples: int) -> list[pd.Series]:
    rows = rows.copy()
    rows["quality"] = (
        pd.to_numeric(rows["score"], errors="coerce").fillna(0.5).clip(0.05, 1.0)
        * np.sqrt(
            pd.to_numeric(rows["w"], errors="coerce").fillna(1.0).clip(lower=1.0)
            * pd.to_numeric(rows["h"], errors="coerce").fillna(1.0).clip(lower=1.0)
        )
    )
    rows = rows[(rows["w"] >= 8) & (rows["h"] >= 24)].sort_values("frame")
    if rows.empty:
        return []
    sample_count = min(maximum_samples, len(rows))
    selected = []
    for indices in np.array_split(np.arange(len(rows)), sample_count):
        if not len(indices):
            continue
        window = rows.iloc[indices]
        selected.append(window.loc[window["quality"].idxmax()])
    return selected


def _player_crop(image: np.ndarray, row: pd.Series) -> Image.Image | None:
    height, width = image.shape[:2]
    x1 = max(0, int(float(row["x"])))
    y1 = max(0, int(float(row["y"])))
    x2 = min(width, int(float(row["x"]) + float(row["w"])))
    y2 = min(height, int(float(row["y"]) + float(row["h"])))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # Keep the head and torso while reducing grass and shorts. The crop remains
    # large enough for the football-domain CLIP color projection.
    crop_height, crop_width = crop.shape[:2]
    torso = crop[: max(4, int(crop_height * 0.68)), int(crop_width * 0.08): max(int(crop_width * 0.92), 1)]
    if torso.size == 0:
        torso = crop
    return Image.fromarray(cv2.cvtColor(torso, cv2.COLOR_BGR2RGB))


def refine_identities_with_clip(
    track_df: pd.DataFrame,
    image_dir: Path,
    decisions: dict[int, TrackIdentity],
    color_groups: dict[str, set[str]],
    model_path: Path,
    maximum_samples_per_track: int = 6,
    batch_size: int = 16,
    minimum_margin: float = 0.06,
    override_color_confidence_below: float = 0.86,
    maximum_anchor_tracks_per_team: int = 40,
) -> tuple[dict[int, TrackIdentity], dict]:
    """Aggregate fine-tuned CLIP color features over identity-consistent tracklets."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not any(color_groups.values()):
        return decisions, {"status": "skipped", "reason": "no_color_groups"}

    model = CLIPFinetune()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model = model.to(device).eval()
    people = track_df[track_df["role"].astype(str).str.lower() != "ball"]
    people_track_ids = {int(value) for value in people["track_id"].unique()}
    anchor_ids: set[int] = set()
    for identity in ("team0", "team1"):
        candidates = sorted(
            (
                decision
                for decision in decisions.values()
                if decision.identity == identity
                and decision.confidence >= 0.90
                and decision.evidence_frames >= 3
            ),
            key=lambda decision: (decision.confidence, decision.evidence_frames),
            reverse=True,
        )
        anchor_ids.update(
            int(decision.track_id)
            for decision in candidates[: max(1, maximum_anchor_tracks_per_team)]
        )
    unresolved_ids = people_track_ids - set(decisions)
    override_candidate_ids = {
        int(track_id)
        for track_id, decision in decisions.items()
        if decision.identity in {"team0", "team1"}
        and decision.confidence < override_color_confidence_below
    }
    target_track_ids = anchor_ids | unresolved_ids | override_candidate_ids

    samples: list[tuple[int, Image.Image]] = []
    sampled_by_track: dict[int, int] = defaultdict(int)
    for raw_track_id, rows in people.groupby("track_id", sort=False):
        if int(raw_track_id) not in target_track_ids:
            continue
        for row in _sample_rows(rows, maximum_samples_per_track):
            image = cv2.imread(str(image_dir / f"{int(row['frame']):06d}.jpg"))
            if image is None:
                continue
            crop = _player_crop(image, row)
            if crop is not None:
                track_id = int(raw_track_id)
                samples.append((track_id, crop))
                sampled_by_track[track_id] += 1

    features_by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(samples), max(1, batch_size)):
            batch = samples[start: start + max(1, batch_size)]
            images = torch.stack([model.preprocess(image) for _, image in batch]).to(device)
            outputs = model(images)
            features = F.normalize(outputs["color_embedding"].float(), p=2, dim=1)
            for (track_id, _), feature in zip(batch, features.cpu().numpy()):
                features_by_track[track_id].append(feature)

    descriptors = {}
    for track_id, feature_rows in features_by_track.items():
        descriptor = np.median(np.stack(feature_rows), axis=0)
        descriptors[track_id] = descriptor / max(float(np.linalg.norm(descriptor)), 1e-6)
    prototypes = {}
    for identity in ("team0", "team1"):
        anchors = [
            descriptors[track_id]
            for track_id, decision in decisions.items()
            if track_id in descriptors
            and decision.identity == identity
            and track_id in anchor_ids
        ]
        if anchors:
            prototype = np.mean(np.stack(anchors), axis=0)
            prototypes[identity] = prototype / max(float(np.linalg.norm(prototype)), 1e-6)
    if set(prototypes) != {"team0", "team1"}:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return decisions, {
            "status": "skipped",
            "reason": "insufficient_same_match_anchors",
            "sampled_tracks": len(descriptors),
            "sampled_crops": len(samples),
        }

    updated = dict(decisions)
    changed = []
    appearance_rows = []
    for track_id, descriptor in descriptors.items():
        identity_scores = {
            identity: float(np.dot(descriptor, prototype))
            for identity, prototype in prototypes.items()
        }
        existing = updated.get(track_id)
        if existing and existing.identity in {"referee", "goalkeeper_team0", "goalkeeper_team1"}:
            appearance_rows.append(
                {
                    "track_id": track_id,
                    "samples": sampled_by_track[track_id],
                    "identity_scores": identity_scores,
                    "decision": existing.identity,
                    "action": "preserved_special_role",
                }
            )
            continue

        team_scores = {key: identity_scores.get(key, 0.0) for key in ("team0", "team1")}
        ranked = sorted(team_scores.items(), key=lambda item: item[1], reverse=True)
        identity, best = ranked[0]
        second = ranked[1][1]
        probabilities = F.softmax(torch.tensor([best, second]) * 20.0, dim=0).numpy()
        confidence = float(probabilities[0])
        margin = float(probabilities[0] - probabilities[1])
        action = "kept_color_vote"
        should_assign = existing is None and margin >= minimum_margin
        should_override = (
            existing is not None
            and existing.identity in {"team0", "team1"}
            and existing.identity != identity
            and existing.confidence < override_color_confidence_below
            and margin >= minimum_margin
        )
        if should_assign or should_override:
            previous = existing.identity if existing else None
            updated[track_id] = TrackIdentity(
                track_id=track_id,
                identity=identity,
                confidence=round(confidence, 4),
                margin=round(margin, 4),
                evidence_frames=sampled_by_track[track_id],
                scores={key: round(value, 4) for key, value in identity_scores.items()},
                method="same_match_clip_tracklet_prototype",
            )
            action = "assigned" if previous is None else "overridden"
            changed.append({"track_id": track_id, "from": previous, "to": identity, "margin": round(margin, 4)})
        appearance_rows.append(
            {
                "track_id": track_id,
                "samples": sampled_by_track[track_id],
                "identity_scores": {key: round(value, 4) for key, value in identity_scores.items()},
                "decision": updated.get(track_id).identity if updated.get(track_id) else None,
                "action": action,
            }
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return updated, {
        "status": "completed",
        "model": str(model_path),
        "method": "same_match_clip_tracklet_prototypes",
        "anchor_tracks": {
            identity: int(sum(
                decision.identity == identity and decision.confidence >= 0.90 and track_id in descriptors
                for track_id, decision in decisions.items()
            ))
            for identity in ("team0", "team1")
        },
        "unresolved_tracks_considered": int(len(unresolved_ids)),
        "override_candidates_considered": int(len(override_candidate_ids)),
        "sampled_tracks": len(descriptors),
        "sampled_crops": len(samples),
        "changed_tracks": changed,
        "tracks": appearance_rows,
    }
