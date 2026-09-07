"""Passing-network summaries derived from visual pass candidates."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from typing import Any

from ..base import TacticalAnalyzer, register_analyzer
from ..models import AnalysisContext, AnalyzerOutput, Evidence, Finding
from .progression import extract_pass_records


def _largest_component_ratio(nodes: set[int], edges: Counter[tuple[int, int]]) -> float:
    if not nodes:
        return 0.0
    adjacency: dict[int, set[int]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    largest = 1
    visited: set[int] = set()
    for node in nodes:
        if node in visited:
            continue
        queue = deque([node])
        visited.add(node)
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        largest = max(largest, size)
    return largest / len(nodes)


def _network_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    edges = Counter((int(item["actor_track_id"]), int(item["target_track_id"])) for item in records)
    nodes = {node for edge in edges for node in edge}
    involvement = Counter()
    for (source, target), weight in edges.items():
        involvement[source] += weight
        involvement[target] += weight
    total = sum(edges.values())
    probabilities = [weight / total for weight in edges.values()] if total else []
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    normalized_entropy = entropy / math.log(len(edges)) if len(edges) > 1 else 0.0
    possible_edges = len(nodes) * (len(nodes) - 1)
    centrality = [
        {"track_id": track_id, "pass_involvement": count, "degree_centrality": round(count / max(2 * total, 1), 4)}
        for track_id, count in involvement.most_common()
    ]
    return {
        "nodes": len(nodes),
        "directed_edges": len(edges),
        "pass_candidates": total,
        "density": round(len(edges) / max(possible_edges, 1), 4),
        "normalized_pass_entropy": round(normalized_entropy, 4),
        "largest_component_ratio": round(_largest_component_ratio(nodes, edges), 4),
        "centrality": centrality,
        "edges": [
            {"source_track_id": source, "target_track_id": target, "weight": weight}
            for (source, target), weight in edges.most_common()
        ],
    }


@register_analyzer("pass_network")
class PassNetworkAnalyzer(TacticalAnalyzer):
    priority = "P2"
    description_zh = "构建传球关系图并计算中心性、连通性和传球熵"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        records = extract_pass_records(context, self.config)
        findings: list[Finding] = []
        teams: dict[str, dict[str, Any]] = {}
        minimum_passes = int(self.config.get("minimum_passes", 3))
        for team in ("left", "right"):
            team_records = [record for record in records if record["team"] == team]
            metrics = _network_metrics(team_records)
            teams[team] = metrics
            if metrics["pass_candidates"] < minimum_passes:
                continue
            label = "左队" if team == "left" else "右队"
            connector = metrics["centrality"][0] if metrics["centrality"] else None
            connector_text = (
                f"跟踪ID {connector['track_id']}参与{connector['pass_involvement']}次候选连接"
                if connector
                else "暂未形成稳定核心节点"
            )
            confidence = min(0.88, 0.45 + metrics["pass_candidates"] / 40.0)
            findings.append(
                Finding(
                    finding_id=f"pass-network-{team}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="pass_network_structure",
                    title_zh=f"{label}传球网络结构",
                    summary_zh=(
                        f"{metrics['nodes']}个节点、{metrics['directed_edges']}条有向连接，"
                        f"归一化传球熵{metrics['normalized_pass_entropy']:.2f}；{connector_text}。"
                    ),
                    confidence=round(confidence, 4),
                    start_sec=context.frames[0].time_sec if context.frames else None,
                    end_sec=context.duration_sec,
                    metrics=metrics,
                    evidence=[Evidence(None, None, "由视觉传球候选聚合形成有向加权网络。", metrics)],
                    limitations_zh=["跟踪ID不等于球衣号码；短片段网络不能代表整场组织核心。"],
                )
            )
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "directed_visual_pass_network_v1", "teams": teams},
            findings=findings,
            warnings=["网络统计只使用确认接球的视觉候选，未识别传球不会进入网络。"],
        )
