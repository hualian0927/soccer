"""Lightweight jersey-color features for distant broadcast player crops."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorEstimate:
    label: str
    confidence: float
    scores: dict[str, float]


def _weighted_ratio(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * mask.astype(np.float32)) / max(np.sum(weights), 1e-6))


def estimate_jersey_color(crop: np.ndarray | None) -> ColorEstimate:
    """Classify the torso color while suppressing grass and crop-border pixels."""
    if crop is None or crop.size == 0:
        return ColorEstimate("unknown", 0.0, {})

    height, width = crop.shape[:2]
    if height < 4 or width < 2:
        return ColorEstimate("unknown", 0.0, {})

    # The lower body, shorts and surrounding pitch are common causes of blue/white swaps.
    y1, y2 = int(height * 0.08), max(int(height * 0.56), 1)
    x1, x2 = int(width * 0.18), max(int(width * 0.82), 1)
    torso = crop[y1:y2, x1:x2]
    if torso.size == 0:
        torso = crop
    torso = cv2.resize(torso, (48, 48), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    yy, xx = np.mgrid[0:48, 0:48]
    center_weight = np.exp(-(((xx - 23.5) / 15.0) ** 2 + ((yy - 21.0) / 18.0) ** 2))
    center_weight *= 0.35 + 0.65 * (yy <= 38)

    masks = {
        "white": (saturation < 72) & (value > 142),
        "black": value < 68,
        "grey": (saturation < 48) & (value >= 68) & (value <= 155),
        "red": ((hue <= 11) | (hue >= 169)) & (saturation > 55) & (value > 48),
        "orange": (hue >= 12) & (hue < 21) & (saturation > 62) & (value > 65),
        "yellow": (hue >= 21) & (hue <= 38) & (saturation > 52) & (value > 72),
        "green": (hue >= 39) & (hue <= 88) & (saturation > 42) & (value > 42),
        # Distant light-blue shirts lose saturation after resize and compression.
        "blue": (hue >= 89) & (hue <= 136) & (saturation > 24) & (value > 48),
        "purple": (hue >= 137) & (hue <= 160) & (saturation > 38) & (value > 48),
        "pink": (hue > 160) & (hue < 169) & (saturation > 38) & (value > 72),
    }
    scores = {name: _weighted_ratio(mask, center_weight) for name, mask in masks.items()}

    # Estimate crop-border background. A green border is usually pitch, not jersey.
    full_hsv = cv2.cvtColor(cv2.resize(crop, (48, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    border = np.concatenate(
        [full_hsv[:, :7].reshape(-1, 3), full_hsv[:, -7:].reshape(-1, 3)],
        axis=0,
    )
    border_green = np.mean(
        (border[:, 0] >= 39)
        & (border[:, 0] <= 88)
        & (border[:, 1] > 38)
        & (border[:, 2] > 38)
    )
    if border_green > 0.35:
        scores["green"] *= max(0.18, 1.0 - float(border_green) * 0.9)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    label, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    minimum_score = 0.055 if label in {"blue", "red", "yellow", "green"} else 0.075
    if best < minimum_score:
        return ColorEstimate("unknown", 0.0, {name: round(score, 4) for name, score in scores.items()})
    confidence = min(1.0, 0.45 * best / max(minimum_score, 1e-6) + 0.55 * max(0.0, best - second) / max(best, 1e-6))
    return ColorEstimate(label, round(confidence, 4), {name: round(score, 4) for name, score in scores.items()})
