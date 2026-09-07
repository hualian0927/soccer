"""P1 formation tendency based on long-range aggregation of valid far-view frames."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..base import TacticalAnalyzer, register_analyzer
from ..models import AnalysisContext, AnalyzerOutput, Evidence, Finding, FrameState
from .events import attack_direction


FORMATION_TEMPLATES = {
    "4-3-3": [4, 3, 3],
    "4-4-2": [4, 4, 2],
    "3-5-2": [3, 5, 2],
    "3-4-3": [3, 4, 3],
    "5-3-2": [5, 3, 2],
    "5-4-1": [5, 4, 1],
    "4-2-3-1": [4, 2, 3, 1],
    "4-1-4-1": [4, 1, 4, 1],
}


def kmeans_1d(values: np.ndarray, clusters: int, iterations: int = 20) -> tuple[np.ndarray, np.ndarray, float]:
    centers = np.quantile(values, np.linspace(0.08, 0.92, clusters)).astype(np.float32)
    labels = np.zeros(len(values), dtype=np.int32)
    for _ in range(iterations):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1).astype(np.int32)
        updated = centers.copy()
        for index in range(clusters):
            members = values[labels == index]
            if len(members):
                updated[index] = float(members.mean())
        if np.allclose(updated, centers):
            break
        centers = updated
    order = np.argsort(centers)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    ordered_labels = np.array([remap[int(label)] for label in labels], dtype=np.int32)
    ordered_centers = centers[order]
    error = float(np.mean((values - ordered_centers[ordered_labels]) ** 2))
    return ordered_labels, ordered_centers, error


def scale_counts(counts: list[int], target: int = 10) -> list[int]:
    values = np.asarray(counts, dtype=np.float32)
    scaled = values / max(float(values.sum()), 1.0) * target
    result = np.floor(scaled).astype(int)
    result = np.maximum(result, 1)
    while int(result.sum()) < target:
        index = int(np.argmax(scaled - result))
        result[index] += 1
    while int(result.sum()) > target:
        candidates = [index for index, value in enumerate(result) if value > 1]
        if not candidates:
            break
        index = max(candidates, key=lambda item: result[item] - scaled[item])
        result[index] -= 1
    return [int(value) for value in result]


def template_distance(counts: list[int], template: list[int]) -> float:
    left = np.asarray(counts, dtype=np.float32) / max(sum(counts), 1)
    right = np.asarray(template, dtype=np.float32) / max(sum(template), 1)
    return float(np.abs(left - right).sum())


def estimate_frame(frame: FrameState, team: str, min_visible: int) -> dict | None:
    players_by_track = {
        player.track_id: player
        for player in frame.players
        if player.team == team and player.role == "player" and player.has_pitch_position
    }
    players = list(players_by_track.values())
    if len(players) < min_visible:
        return None
    if len(players) > 10:
        players = players[:10]
    direction = attack_direction(frame, team)
    x_values = np.asarray([float(player.pitch_x) for player in players], dtype=np.float32)
    y_values = np.asarray([float(player.pitch_y) for player in players], dtype=np.float32)
    axis = x_values * direction
    width = float(np.ptp(y_values))
    depth = float(np.ptp(x_values))

    candidates = []
    for clusters in (3, 4):
        if len(players) < clusters + 3:
            continue
        labels, centers, error = kmeans_1d(axis, clusters)
        counts = [int(np.sum(labels == index)) for index in range(clusters)]
        if not all(counts):
            continue
        scaled = scale_counts(counts)
        templates = {
            name: template for name, template in FORMATION_TEMPLATES.items() if len(template) == clusters
        }
        name, template = min(templates.items(), key=lambda item: template_distance(scaled, item[1]))
        distance = template_distance(scaled, template)
        separation = float(np.min(np.diff(centers))) if len(centers) > 1 else 0.0
        cluster_penalty = error / max(depth, 1.0)
        score = distance + cluster_penalty * 0.10 + (0.04 if clusters == 4 and len(players) < 9 else 0.0)
        candidates.append((score, name, counts, scaled, distance, separation))
    if not candidates:
        return None
    score, name, counts, scaled, distance, separation = min(candidates, key=lambda item: item[0])
    visible_factor = min(1.0, len(players) / 10.0)
    geometry_factor = min(1.0, 0.55 * depth / 45.0 + 0.45 * width / 42.0)
    confidence = max(0.0, min(0.95, (1.0 - distance / 1.05) * visible_factor * max(0.35, geometry_factor)))
    return {
        "frame": frame.frame,
        "time_sec": round(frame.time_sec, 3),
        "team": team,
        "formation": name,
        "confidence": round(confidence, 4),
        "visible_players": len(players),
        "line_counts": counts,
        "scaled_counts": scaled,
        "width_m": round(width, 3),
        "depth_m": round(depth, 3),
        "line_separation_m": round(separation, 3),
    }


@register_analyzer("formation_tendency")
class FormationTendencyAnalyzer(TacticalAnalyzer):
    priority = "P1"
    description_zh = "在多个完整站位帧上聚合4-3-3、4-2-3-1等阵型倾向"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        min_visible = int(self.config.get("min_visible_players", 7))
        sample_step = max(1, int(self.config.get("sample_step_frames", round(context.fps))))
        minimum_confidence = float(self.config.get("minimum_frame_confidence", 0.28))
        minimum_supporting_samples = int(self.config.get("minimum_supporting_samples", 3))
        minimum_aggregate_confidence = float(self.config.get("minimum_aggregate_confidence", 0.45))
        minimum_vote_ratio = float(self.config.get("minimum_vote_ratio", 0.45))
        estimates: dict[str, list[dict]] = defaultdict(list)

        for frame in context.frames:
            if (frame.frame - context.frames[0].frame) % sample_step:
                continue
            for team in ("left", "right"):
                estimate = estimate_frame(frame, team, min_visible)
                if estimate and estimate["confidence"] >= minimum_confidence:
                    estimates[team].append(estimate)

        summary: dict[str, dict] = {}
        findings: list[Finding] = []
        warnings: list[str] = []
        for team in ("left", "right"):
            team_estimates = estimates[team]
            weighted_votes: Counter = Counter()
            raw_votes: Counter = Counter()
            for estimate in team_estimates:
                weighted_votes[estimate["formation"]] += estimate["confidence"]
                raw_votes[estimate["formation"]] += 1
            if not weighted_votes:
                summary[team] = {"status": "insufficient_evidence", "valid_samples": 0}
                warnings.append(f"{team} 队缺少可见人数足够的远景帧，未输出阵型结论。")
                continue
            formation, winning_weight = weighted_votes.most_common(1)[0]
            total_weight = sum(weighted_votes.values())
            matching = [item for item in team_estimates if item["formation"] == formation]
            representative = max(matching, key=lambda item: item["confidence"])
            vote_ratio = winning_weight / max(total_weight, 1e-6)
            temporal_support = min(1.0, len(matching) / 12.0)
            confidence = min(0.92, 0.35 * vote_ratio + 0.35 * representative["confidence"] + 0.30 * temporal_support)
            if (
                len(matching) < minimum_supporting_samples
                or confidence < minimum_aggregate_confidence
                or vote_ratio < minimum_vote_ratio
            ):
                summary[team] = {
                    "status": "insufficient_evidence",
                    "best_candidate": formation,
                    "confidence": round(confidence, 4),
                    "valid_samples": len(team_estimates),
                    "supporting_samples": len(matching),
                    "vote_ratio": round(vote_ratio, 4),
                    "representative_frame": representative,
                }
                warnings.append(
                    f"{team} 队阵型候选 {formation} 获得 {len(matching)} 个支持样本、"
                    f"投票占比 {vote_ratio:.0%}，未达到输出门槛。"
                )
                if len(matching) >= minimum_supporting_samples:
                    label = "左队" if team == "left" else "右队"
                    findings.append(
                        Finding(
                            finding_id=f"formation-frame-{team}",
                            analyzer=self.name,
                            priority=self.priority,
                            category="formation_candidate",
                            title_zh=f"{label}阵型代表帧：{formation}（低置信）",
                            summary_zh=(
                                f"该帧呈现接近{formation}的可见站位线；全片投票占比仅"
                                f"{vote_ratio:.0%}，因此只作为时刻结构展示。"
                            ),
                            confidence=round(min(confidence, representative["confidence"]), 4),
                            start_sec=representative["time_sec"],
                            end_sec=representative["time_sec"],
                            metrics=summary[team],
                            evidence=[
                                Evidence(
                                    frame=representative["frame"],
                                    time_sec=representative["time_sec"],
                                    description_zh="该候选阵型中完整度和模板匹配度最高的真实帧。",
                                    metrics=representative,
                                )
                            ],
                            limitations_zh=[
                                "该结果只描述代表帧的可见站位，不构成整场固定阵型结论。",
                                "转播切镜、遮挡和攻守阶段变化会降低跨时间投票一致性。",
                            ],
                        )
                    )
                continue
            summary[team] = {
                "status": "candidate",
                "formation": formation,
                "confidence": round(confidence, 4),
                "valid_samples": len(team_estimates),
                "supporting_samples": len(matching),
                "vote_ratio": round(vote_ratio, 4),
                "weighted_votes": {key: round(value, 4) for key, value in weighted_votes.items()},
                "raw_votes": dict(raw_votes),
                "representative_frame": representative,
            }
            label = "左队" if team == "left" else "右队"
            findings.append(
                Finding(
                    finding_id=f"formation-{team}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="formation",
                    title_zh=f"{label}阵型倾向：{formation}",
                    summary_zh=(
                        f"在{len(team_estimates)}个有效远景采样中，{len(matching)}个支持该阵型；"
                        f"代表帧线型为{representative['scaled_counts']}。"
                    ),
                    confidence=round(confidence, 4),
                    start_sec=context.frames[0].time_sec if context.frames else None,
                    end_sec=context.duration_sec,
                    metrics=summary[team],
                    evidence=[
                        Evidence(
                            frame=representative["frame"],
                            time_sec=representative["time_sec"],
                            description_zh="完整度和模板匹配度最高的代表帧。",
                            metrics=representative,
                        )
                    ],
                    limitations_zh=[
                        "这是转播画面可见球员的阵型倾向，不能替代全场光学追踪数据。",
                        "阵型会随攻守阶段变化，报告采用跨时间主导结构而非固定标签。",
                    ],
                )
            )

        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"teams": summary, "sampling_step_frames": sample_step},
            findings=findings,
            warnings=warnings,
        )
