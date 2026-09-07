"""Machine-readable and coach-readable output writers."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .models import AnalysisReport, Finding
from .l1 import write_l1_bundle


def write_report_bundle(report: AnalysisReport, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tactical_analysis_report.json"
    markdown_path = output_dir / "tactical_analysis_report.md"
    timeline_path = output_dir / "event_timeline.csv"
    summary_path = output_dir / "match_summary.json"
    highlights_path = output_dir / "highlight_manifest.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    write_timeline(report, timeline_path)
    summary = build_match_summary(report)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    highlights = build_highlight_manifest(report)
    write_highlights(highlights, highlights_path)
    paths = {
        "json": json_path,
        "markdown": markdown_path,
        "timeline": timeline_path,
        "summary": summary_path,
        "highlights": highlights_path,
    }
    paths.update(write_l1_bundle(report, output_dir))
    return paths


def analyzer_summary(report: AnalysisReport, name: str) -> dict[str, Any]:
    output = next((item for item in report.analyzer_outputs if item.analyzer == name), None)
    return output.summary if output else {}


def build_match_summary(report: AnalysisReport) -> dict[str, Any]:
    event_counts = Counter(event.event_type for event in report.events)
    team_events: dict[str, Counter] = {"left": Counter(), "right": Counter()}
    player_events: dict[str, Counter] = defaultdict(Counter)
    shots = []
    for event in report.events:
        if event.team in team_events:
            team_events[event.team][event.event_type] += 1
        if event.actor_track_id is not None:
            player_events[str(event.actor_track_id)][event.event_type] += 1
        if event.event_type == "shot_candidate":
            shots.append(
                {
                    "time_sec": event.time_sec,
                    "team": event.team,
                    "actor_track_id": event.actor_track_id,
                    "confidence": event.confidence,
                    **event.metrics,
                }
            )
    timeline = analyzer_summary(report, "event_timeline")
    spatial = analyzer_summary(report, "spatial_structure").get("teams", {})
    phases = analyzer_summary(report, "phase_of_play").get("teams", {})
    shapes = analyzer_summary(report, "team_shape_engine").get("teams", {})
    set_pieces = analyzer_summary(report, "set_piece_delivery").get("teams", {})
    goalkeepers = analyzer_summary(report, "goalkeeper_interventions").get("teams", {})
    progression = analyzer_summary(report, "progression_analysis").get("teams", {})
    chains = analyzer_summary(report, "attacking_chains").get("teams", {})
    defensive_risk = analyzer_summary(report, "defensive_third_risk").get("teams", {})
    numerical_superiority = analyzer_summary(report, "numerical_superiority").get("teams", {})
    defensive_gaps = analyzer_summary(report, "defensive_gaps").get("teams", {})
    pressing = analyzer_summary(report, "pressing_analysis").get("teams", {})
    transitions = analyzer_summary(report, "transition_analysis").get("teams", {})
    attacking_transitions = analyzer_summary(report, "transition_analysis").get("attacking_teams", {})
    passing_networks = analyzer_summary(report, "pass_network").get("teams", {})
    return {
        "schema_version": "1.2.0",
        "source": report.source,
        "data_scope": "pure_visual_candidate_metrics",
        "event_counts": dict(event_counts),
        "possession_proxy": timeline.get("possession_proxy", {}),
        "shooting_structure": timeline.get("shooting_structure", {}),
        "priority_coverage": {
            "P0": ["event_timeline", "automatic_highlights", "shooting_structure", "spatial_zones", "match_summary"],
            "P1": ["average_positions", "formation", "team_shape", "set_piece_landings", "goalkeeper_interventions"],
            "P2": ["pass_direction_distance", "pass_types", "carries", "attacking_chains", "defensive_third_risk"],
            "P3": ["numerical_superiority", "high_press_ppda", "defensive_gaps", "transition_speed"],
        },
        "teams": {
            team: {
                "events": dict(team_events[team]),
                "spatial_structure": spatial.get(team, {}),
                "phase_of_play": phases.get(team, {}),
                "team_shape": shapes.get(team, {}),
                "set_pieces": set_pieces.get(team, {}),
                "goalkeeper": goalkeepers.get(team, {}),
                "progression": progression.get(team, {}),
                "attacking_chains": chains.get(team, {}),
                "defensive_third_risk": defensive_risk.get(team, {}),
                "numerical_superiority": numerical_superiority.get(team, {}),
                "defensive_gaps": defensive_gaps.get(team, {}),
                "pass_network": passing_networks.get(team, {}),
                "pressing": pressing.get(team, {}),
                "defensive_transition": transitions.get(team, {}),
                "attacking_transition": attacking_transitions.get(team, {}),
            }
            for team in ("left", "right")
        },
        "shots": shots,
        "players": {
            track_id: dict(counts)
            for track_id, counts in sorted(
                player_events.items(),
                key=lambda item: sum(item[1].values()),
                reverse=True,
            )
        },
        "interpretation_limits_zh": [
            "射门、传球、触球和推进均为视觉候选，未接入官方比赛事件结果。",
            "没有可靠球门结果标签时不统计进球、射正率或扑救成功率真值。",
            "球队结构只基于转播画面中可见且成功映射到球场坐标的球员。",
        ],
    }


HIGHLIGHT_WEIGHTS = {
    "shot_candidate": 1.00,
    "corner_candidate": 0.82,
    "line_breaking_pass_candidate": 0.86,
    "progressive_pass_candidate": 0.72,
    "attacking_transition_candidate": 0.82,
    "defensive_transition_candidate": 0.72,
    "pressing_episode_candidate": 0.78,
    "local_overload_candidate": 0.66,
    "carry_candidate": 0.65,
    "set_piece_delivery_candidate": 0.82,
    "set_piece_landing_candidate": 0.82,
    "goalkeeper_intervention_candidate": 0.88,
    "goalkeeper_save_candidate": 0.88,
    "attacking_chain_candidate": 0.90,
    "defensive_risk_candidate": 0.76,
    "defensive_third_risk": 0.74,
    "numerical_superiority_candidate": 0.84,
    "high_press_candidate": 0.84,
    "high_press_ppda_summary": 0.75,
    "defensive_gap_candidate": 0.82,
    "transition_speed_candidate": 0.84,
}

HIGHLIGHT_TAG_LABELS_ZH = {
    "shot_candidate": "射门",
    "corner_candidate": "角球",
    "line_breaking_pass_candidate": "穿线传球",
    "progressive_pass_candidate": "推进传球",
    "attacking_transition_candidate": "进攻转换",
    "defensive_transition_candidate": "防守转换",
    "pressing_episode_candidate": "压迫",
    "local_overload_candidate": "局部人数优势",
    "carry_candidate": "持球推进",
    "set_piece_delivery_candidate": "定位球",
    "set_piece_landing_candidate": "定位球落点",
    "goalkeeper_intervention_candidate": "门将干预",
    "goalkeeper_save_candidate": "门将扑救",
    "attacking_chain_candidate": "进攻链",
    "defensive_risk_candidate": "防守风险",
    "defensive_third_risk": "防守三区风险",
    "numerical_superiority_candidate": "动态多打少",
    "high_press_candidate": "高位逼抢",
    "high_press_ppda_summary": "PPDA",
    "defensive_gap_candidate": "防守空档",
    "transition_speed_candidate": "转换速度",
}


def format_highlight_tags_zh(raw_tags: str, limit: int = 5) -> str:
    tags = list(dict.fromkeys(tag for tag in raw_tags.split("|") if tag))
    labels = [HIGHLIGHT_TAG_LABELS_ZH.get(tag, tag) for tag in tags]
    visible = labels[:limit]
    suffix = f"等{len(labels)}类" if len(labels) > limit else ""
    return "、".join(visible) + suffix


def build_highlight_manifest(report: AnalysisReport) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for event in report.events:
        weight = HIGHLIGHT_WEIGHTS.get(event.event_type)
        if weight is None:
            continue
        pre, post = (5.0, 4.0) if event.event_type == "shot_candidate" else (4.0, 4.0)
        candidates.append(
            {
                "type": event.event_type,
                "label_zh": event.label_zh,
                "team": event.team,
                "center_sec": event.time_sec,
                "start_sec": max(0.0, event.time_sec - pre),
                "end_sec": min(float(report.source.get("duration_sec", event.time_sec + post)), event.time_sec + post),
                "score": round(weight * 0.65 + event.confidence * 0.35, 4),
                "confidence": event.confidence,
                "evidence_frame": event.frame,
                "source_id": event.event_id,
            }
        )
    for finding in report.findings:
        weight = HIGHLIGHT_WEIGHTS.get(finding.category)
        if weight is None or finding.start_sec is None:
            continue
        center = next(
            (item.time_sec for item in finding.evidence if item.time_sec is not None),
            (finding.start_sec + (finding.end_sec or finding.start_sec)) / 2.0,
        )
        candidates.append(
            {
                "type": finding.category,
                "label_zh": finding.title_zh,
                "team": None,
                "center_sec": round(float(center), 3),
                "start_sec": max(0.0, finding.start_sec),
                "end_sec": min(float(report.source.get("duration_sec", finding.end_sec or center)), finding.end_sec or center),
                "score": round(weight * 0.65 + finding.confidence * 0.35, 4),
                "confidence": finding.confidence,
                "evidence_frame": next((item.frame for item in finding.evidence if item.frame is not None), None),
                "source_id": finding.finding_id,
            }
        )
    selected = []
    for candidate in sorted(candidates, key=lambda item: (-item["score"], item["center_sec"])):
        duplicate = next(
            (item for item in selected if abs(candidate["center_sec"] - item["center_sec"]) < 3.0),
            None,
        )
        if duplicate is not None:
            duplicate.setdefault("analysis_tags", []).append(candidate["type"])
            continue
        if len(selected) >= 24:
            continue
        candidate["analysis_tags"] = [candidate["type"]]
        selected.append(candidate)
    selected.sort(key=lambda item: item["center_sec"])
    for index, item in enumerate(selected, 1):
        item["highlight_id"] = f"highlight-{index:03d}"
        item["duration_sec"] = round(max(0.0, item["end_sec"] - item["start_sec"]), 3)
        item["analysis_tags"] = "|".join(dict.fromkeys(item["analysis_tags"]))
    return selected


def write_highlights(highlights: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "highlight_id", "type", "analysis_tags", "label_zh", "team", "center_sec", "start_sec", "end_sec",
        "duration_sec", "score", "confidence", "evidence_frame", "source_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(highlights)


def write_timeline(report: AnalysisReport, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_id",
                "time_sec",
                "frame",
                "event_type",
                "label_zh",
                "team",
                "actor_track_id",
                "target_track_id",
                "confidence",
                "metrics_json",
            ],
        )
        writer.writeheader()
        for event in sorted(report.events, key=lambda item: (item.time_sec, item.event_id)):
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "time_sec": event.time_sec,
                    "frame": event.frame,
                    "event_type": event.event_type,
                    "label_zh": event.label_zh,
                    "team": event.team,
                    "actor_track_id": event.actor_track_id,
                    "target_track_id": event.target_track_id,
                    "confidence": event.confidence,
                    "metrics_json": json.dumps(event.metrics, ensure_ascii=False),
                }
            )


def render_markdown(report: AnalysisReport) -> str:
    source = report.source
    lines = [
        "# 足球技战术分析报告",
        "",
        f"- GSR 数据：`{source['gsr_json']}`",
        f"- 视频：`{source.get('video') or '未提供'}`",
        f"- 时长：{source.get('duration_sec', 0):.2f} 秒",
        f"- 分析帧：{source.get('observed_frame_count', 0)}",
        "",
    ]
    match_summary = build_match_summary(report)
    highlights = build_highlight_manifest(report)
    event_counts = match_summary["event_counts"]
    lines.extend(
        [
            "## 比赛概览",
            "",
            f"- 稳定触球候选：{event_counts.get('touch_candidate', 0)}",
            f"- 接球 / 传球候选：{event_counts.get('receive_candidate', 0)} / {event_counts.get('pass_candidate', 0)}",
            f"- 持球推进候选：{event_counts.get('carry_candidate', 0)}",
            f"- 射门 / 角球候选：{event_counts.get('shot_candidate', 0)} / {event_counts.get('corner_candidate', 0)}",
            f"- 定位球落点 / 门将干预候选：{event_counts.get('set_piece_delivery_candidate', 0)} / {event_counts.get('goalkeeper_intervention_candidate', 0)}",
            f"- 射门进攻链 / 高风险防守片段：{event_counts.get('attacking_chain_candidate', 0)} / {event_counts.get('defensive_risk_candidate', 0)}",
            f"- P3 多打少 / 高位逼抢 / 防守空档 / 转换速度：{event_counts.get('numerical_superiority_candidate', 0)} / {event_counts.get('high_press_candidate', 0)} / {event_counts.get('defensive_gap_candidate', 0)} / {event_counts.get('transition_speed_candidate', 0)}",
            "",
        ]
    )
    lines.extend(["## P0-P3 球队级结果", ""])
    for team, label in (("left", "左队"), ("right", "右队")):
        team_data = match_summary["teams"][team]
        shooting = match_summary["shooting_structure"].get(team, {})
        progress = team_data.get("progression", {})
        set_piece = team_data.get("set_pieces", {})
        goalkeeper = team_data.get("goalkeeper", {})
        chains = team_data.get("attacking_chains", {})
        risk = team_data.get("defensive_third_risk", {})
        superiority = team_data.get("numerical_superiority", {})
        pressing = team_data.get("pressing", {})
        gaps = team_data.get("defensive_gaps", {})
        transition = team_data.get("attacking_transition", {})
        pass_types = progress.get("pass_type_counts", {})
        lines.extend(
            [
                f"### {label}",
                "",
                f"- P0 射门结构：{shooting.get('shot_candidates', 0)} 次候选，禁区内 {shooting.get('in_penalty_area', 0)} 次，平均距离 {float(shooting.get('mean_distance_m') or 0):.1f} 米。",
                f"- P1 定位球 / 门将干预：{set_piece.get('deliveries', 0)} / {goalkeeper.get('intervention_candidates', 0)} 次候选。",
                f"- P2 传球类型：短传 {pass_types.get('short_pass', 0)}、长传 {pass_types.get('long_pass', 0)}、直塞 {pass_types.get('through_ball', 0)}、传中 {pass_types.get('cross', 0)}。",
                f"- P2 进攻链：{chains.get('shot_ending_chains', 0)} 条；主要防守风险通道：{risk.get('highest_risk_channel') or '证据不足'}。",
                f"- P3 多打少 / 防守空档：{superiority.get('episodes', 0)} / {gaps.get('gap_episodes', 0)} 个候选；高位压迫 {pressing.get('zone_counts', {}).get('high_press', 0)} 个。",
                f"- P3 PPDA / 平均转换速度：{pressing.get('ppda_proxy') if pressing.get('ppda_proxy') is not None else '样本不足'} / {float(transition.get('mean_transition_speed_mps') or 0):.2f} m/s。",
                "",
            ]
        )
    lines.extend(["## 推荐复核片段", ""])
    for item in sorted(highlights, key=lambda value: -value["score"])[:8]:
        readable_tags = format_highlight_tags_zh(item["analysis_tags"])
        lines.append(
            f"- {item['start_sec']:.2f}-{item['end_sec']:.2f}秒：{item['label_zh']}"
            f"（标签 {readable_tags}，排序分 {item['score']:.2f}）"
        )
    lines.extend(["", "## 分析结论", ""])
    findings = sorted(
        select_summary_findings(report),
        key=lambda item: (int(item.priority[1:]) if item.priority[1:].isdigit() else 9, -item.confidence),
    )
    if not findings:
        lines.append("本次没有形成满足证据门槛的结论。")
    for finding in findings:
        time_text = ""
        if finding.start_sec is not None:
            time_text = f"（{finding.start_sec:.2f}-{finding.end_sec or finding.start_sec:.2f}秒）"
        lines.extend(
            [
                f"### [{finding.priority}] {finding.title_zh}{time_text}",
                "",
                f"{finding.summary_zh} 置信度：{finding.confidence:.0%}。",
                "",
            ]
        )

    lines.extend(["## 模块状态", ""])
    for output in report.analyzer_outputs:
        lines.append(
            f"- `{output.analyzer}`（{output.priority}）："
            f"{len(output.findings)} 条结论，{len(output.events)} 个时间轴事件"
        )
        for warning in output.warnings:
            lines.append(f"  - 注意：{warning}")

    lines.extend(["", "## 使用边界", ""])
    lines.extend(f"- {item}" for item in report.limitations_zh)
    lines.extend(
        [
            "",
            "## 下一步可视化",
            "",
            "事件时间轴可用于截取候选片段；远景结果可交给阵型/战术视频脚本，近景射门片段可交给个人姿态与门将模块。",
            "",
        ]
    )
    return "\n".join(lines)


def select_summary_findings(report: AnalysisReport) -> list[Finding]:
    """Keep the readable report concise while preserving every event in JSON/CSV."""
    selected = []
    category_counts: dict[str, int] = {}
    category_times: dict[str, list[float]] = {}
    event_categories = {
        "pass_candidate",
        "possession_change",
        "shot_candidate",
        "corner_candidate",
        "local_overload_candidate",
        "corner_structure_candidate",
        "progressive_pass_candidate",
        "line_breaking_pass_candidate",
        "attacking_transition_candidate",
        "attacking_chain_candidate",
        "defensive_third_risk",
        "set_piece_landing_candidate",
        "goalkeeper_save_candidate",
        "numerical_superiority_candidate",
        "high_press_ppda_summary",
        "defensive_gap_candidate",
        "transition_speed_candidate",
        "pressing_episode_candidate",
    }
    ranked = sorted(report.findings, key=lambda item: -item.confidence)
    for finding in ranked:
        if finding.confidence < 0.45 and finding.category != "data_quality":
            continue
        limit = 2 if finding.category in event_categories else 4
        count = category_counts.get(finding.category, 0)
        if count >= limit:
            continue
        if finding.category in event_categories and finding.start_sec is not None:
            center = (finding.start_sec + (finding.end_sec or finding.start_sec)) / 2.0
            if any(abs(center - old) < 2.0 for old in category_times.get(finding.category, [])):
                continue
            category_times.setdefault(finding.category, []).append(center)
        selected.append(finding)
        category_counts[finding.category] = count + 1
    return selected[:20]
