"""Unified L1 event catalog derived from evidence-first analyzer outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .models import AnalysisReport, TimelineEvent


EVENT_PRESENTATION = {
    "shot_candidate": ("L1-01", "射门", "shot", "orange"),
    "corner_candidate": ("L1-02", "定位球", "set_piece", "yellow"),
    "set_piece_delivery_candidate": ("L1-02", "定位球", "set_piece", "yellow"),
    "touch_candidate": ("L1-03", "传接带", "touch", "blue"),
    "receive_candidate": ("L1-03", "传接带", "receive", "blue"),
    "carry_candidate": ("L1-03", "传接带", "carry", "green"),
    "pass_candidate": ("L1-03", "传接带", "pass", "blue"),
    "possession_change": ("L1-04", "球权转换", "possession_change", "cyan"),
    "defensive_intervention_candidate": ("L1-05", "防守干预", "defensive_intervention", "red"),
    "goalkeeper_intervention_candidate": ("L1-06", "门将事件", "goalkeeper_action", "violet"),
}

SET_PIECE_LABELS = {
    "corner": "角球",
    "free_kick": "任意球",
    "goal_kick": "球门球",
    "throw_in": "界外球",
    "kickoff": "中圈开球",
    "penalty": "点球",
    "unknown": "未分类定位球",
}

ZONE_LABELS = {
    "out_of_bounds": "场外区域",
    "penalty_area": "禁区",
    "attacking_half_space": "进攻肋部",
    "wide_channel": "边路",
    "central_channel": "中路",
}

OUTCOME_LABELS = {
    "retained": "开球队控制落点",
    "opponent_control": "对方控制落点",
    "unknown": "落点控制待确认",
}


def _default_window(event_type: str, time_sec: float) -> tuple[float, float]:
    pre_post = {
        "shot_candidate": (3.0, 5.0),
        "corner_candidate": (4.0, 8.0),
        "set_piece_delivery_candidate": (4.0, 8.0),
        "carry_candidate": (1.0, 2.0),
        "goalkeeper_intervention_candidate": (3.0, 5.0),
    }
    pre, post = pre_post.get(event_type, (2.0, 3.0))
    return max(0.0, time_sec - pre), time_sec + post


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _event_subtype(event: TimelineEvent) -> str:
    if event.event_subtype:
        return event.event_subtype
    metrics = event.metrics
    if event.event_type == "set_piece_delivery_candidate":
        return str(metrics.get("set_piece_type") or "unknown")
    if event.event_type == "corner_candidate":
        return "corner"
    if event.event_type == "goalkeeper_intervention_candidate":
        return str(metrics.get("action_type") or "unknown")
    if event.event_type == "defensive_intervention_candidate":
        return str(metrics.get("intervention_subtype") or "unknown")
    return event.event_type.removesuffix("_candidate")


def _event_outcome(event: TimelineEvent) -> str:
    if event.outcome != "unknown":
        return event.outcome
    metrics = event.metrics
    if event.event_type == "set_piece_delivery_candidate":
        if metrics.get("retained_by_taking_team"):
            return "retained"
        if metrics.get("landing_team"):
            return "opponent_control"
    if event.event_type in {"pass_candidate", "receive_candidate", "carry_candidate"}:
        return "success"
    if event.event_type == "goalkeeper_intervention_candidate" and metrics.get("goalkeeper_controlled_after_shot"):
        return "controlled"
    return "unknown"


def _summary_text(event: TimelineEvent, subtype: str, outcome: str) -> str:
    metrics = event.metrics
    if event.event_type in {"corner_candidate", "set_piece_delivery_candidate"}:
        label = SET_PIECE_LABELS.get(subtype, SET_PIECE_LABELS["unknown"])
        raw_zone = str(metrics.get("landing_zone") or "")
        zone = ZONE_LABELS.get(raw_zone, "落点待确认")
        followup = "，10秒内形成射门候选" if metrics.get("shot_within_10s") else ""
        return f"{label}开出，落点区域 {zone}，{OUTCOME_LABELS.get(outcome, '结果待确认')}{followup}"
    if event.event_type == "shot_candidate":
        return f"射门位置与方向满足视觉候选，结果 {outcome}"
    if event.event_type == "pass_candidate":
        return f"同队稳定持球人切换，足球位移 {float(metrics.get('ball_displacement_m') or 0):.1f} 米"
    if event.event_type == "carry_candidate":
        return f"持球推进 {float(metrics.get('forward_progress_m') or 0):.1f} 米"
    if event.event_type == "possession_change":
        return "稳定控球状态跨队切换"
    if event.event_type == "defensive_intervention_candidate":
        return "防守方完成稳定夺回，具体动作类型待视频复核"
    if event.event_type == "goalkeeper_intervention_candidate":
        return f"门将动作 {subtype}，结果 {outcome}"
    return event.label_zh


def build_l1_catalog(report: AnalysisReport) -> dict[str, Any]:
    duration = float(report.source.get("duration_sec") or 0.0)
    set_piece_times = [
        event.time_sec for event in report.events if event.event_type == "set_piece_delivery_candidate"
    ]
    records: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    for output in report.analyzer_outputs:
        for event in output.events:
            presentation = EVENT_PRESENTATION.get(event.event_type)
            if presentation is None:
                continue
            if event.event_type == "corner_candidate" and any(abs(event.time_sec - value) < 3.0 for value in set_piece_times):
                continue
            group_id, group_label, canonical_type, color = presentation
            subtype = _event_subtype(event)
            outcome = _event_outcome(event)
            fallback_start, fallback_end = _default_window(event.event_type, event.time_sec)
            evidence_start = event.evidence_start_sec if event.evidence_start_sec is not None else fallback_start
            evidence_end = event.evidence_end_sec if event.evidence_end_sec is not None else fallback_end
            if event.event_type == "set_piece_delivery_candidate":
                evidence_end = max(evidence_end, float(event.metrics.get("landing_sec") or event.time_sec) + 4.0)
            record = {
                "event_id": f"{output.analyzer}:{event.event_id}",
                "l1_group": group_id,
                "group_label_zh": group_label,
                "event_type": canonical_type,
                "event_subtype": subtype,
                "label_zh": event.label_zh,
                "period": event.period,
                "timestamp_sec": round(event.time_sec, 3),
                "frame": event.frame,
                "team_id": event.team,
                "actor_track_id": event.actor_track_id,
                "recipient_track_id": event.target_track_id,
                "start_x": _first_not_none(event.start_x, event.metrics.get("origin_x"), event.metrics.get("start_x"), event.metrics.get("shot_x")),
                "start_y": _first_not_none(event.start_y, event.metrics.get("origin_y"), event.metrics.get("start_y"), event.metrics.get("shot_y")),
                "end_x": _first_not_none(event.end_x, event.metrics.get("landing_x"), event.metrics.get("end_x")),
                "end_y": _first_not_none(event.end_y, event.metrics.get("landing_y"), event.metrics.get("end_y")),
                "outcome": outcome,
                "confidence": round(event.confidence, 4),
                "evidence_start_sec": round(max(0.0, evidence_start), 3),
                "evidence_end_sec": round(min(duration or evidence_end, evidence_end), 3),
                "source": event.source,
                "review_status": event.review_status,
                "related_event_ids": event.related_event_ids,
                "color": color,
                "summary_zh": _summary_text(event, subtype, outcome),
                "attributes": event.metrics,
            }
            records.append(record)
            group_counts[group_id] += 1
            if group_id == "L1-02":
                subtype_counts[subtype] += 1
    records.sort(key=lambda item: (item["timestamp_sec"], item["event_id"]))
    set_pieces = [item for item in records if item["l1_group"] == "L1-02"]
    return {
        "schema_version": "1.0.0",
        "source": report.source,
        "scope": "L1 evidence-first visual event candidates",
        "summary": {
            "total_events": len(records),
            "group_counts": dict(group_counts),
            "set_piece_type_counts": dict(subtype_counts),
            "set_piece_total": len(set_pieces),
            "confirmed_events": sum(item["review_status"] == "confirmed" for item in records),
            "candidate_events": sum(item["review_status"] == "candidate" for item in records),
        },
        "events": records,
        "set_pieces": set_pieces,
        "limitations_zh": [
            "当前事件来自纯视觉候选，未接入官方比赛事件真值。",
            "定位球落点优先表示首次稳定控制位置，不等同真实三维第一落地点。",
            "低置信事件必须通过对应视频证据复核。",
        ],
    }


def write_l1_bundle(report: AnalysisReport, output_dir: Path) -> dict[str, Path]:
    catalog = build_l1_catalog(report)
    timeline_json = output_dir / "l1_event_timeline.json"
    timeline_csv = output_dir / "l1_event_timeline.csv"
    summary_json = output_dir / "l1_summary.json"
    set_piece_manifest = output_dir / "set_piece_manifest.csv"
    timeline_json.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(catalog["summary"], ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "event_id", "l1_group", "group_label_zh", "event_type", "event_subtype", "label_zh", "period",
        "timestamp_sec", "frame", "team_id", "actor_track_id", "recipient_track_id", "start_x", "start_y",
        "end_x", "end_y", "outcome", "confidence", "evidence_start_sec", "evidence_end_sec", "source",
        "review_status", "related_event_ids", "summary_zh", "attributes_json",
    ]
    with timeline_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in catalog["events"]:
            row = {key: event.get(key) for key in fields}
            row["related_event_ids"] = "|".join(event["related_event_ids"])
            row["attributes_json"] = json.dumps(event["attributes"], ensure_ascii=False)
            writer.writerow(row)

    set_piece_fields = [
        "event_id", "event_subtype", "label_zh", "timestamp_sec", "evidence_start_sec", "evidence_end_sec",
        "team_id", "actor_track_id", "recipient_track_id", "start_x", "start_y", "end_x", "end_y",
        "outcome", "confidence", "summary_zh",
    ]
    with set_piece_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=set_piece_fields)
        writer.writeheader()
        writer.writerows({key: event.get(key) for key in set_piece_fields} for event in catalog["set_pieces"])
    return {
        "l1_timeline_json": timeline_json,
        "l1_timeline_csv": timeline_csv,
        "l1_summary": summary_json,
        "set_piece_manifest": set_piece_manifest,
    }
