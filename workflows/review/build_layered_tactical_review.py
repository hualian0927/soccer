"""Build an evidence-linked L1-L4 bundle and optionally review sampled L1 events."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from workflows.review.review_tactical_candidates_with_vision import (
    REVIEW_PROMPT_VERSION, build_request, call_vision_api, extract_frames, fusion_status, save_evidence, validate_review,
)
from tactical_analysis.gsr_io import parse_frame


PRESENTATION = {
    "shot_candidate": ("射门", "射门候选", "orange", "L1-01"),
    "corner_candidate": ("定位球", "角球候选", "yellow", "L1-02"),
    "set_piece_delivery_candidate": ("定位球", "定位球候选", "yellow", "L1-02"),
    "ball_out_of_play_candidate": ("出界", "出界候选", "yellow", "L1-02"),
    "pass_candidate": ("传接带", "传球候选", "blue", "L1-03"),
    "carry_candidate": ("传接带", "持球推进", "green", "L1-03"),
    "defensive_intervention_candidate": ("防守干预", "防守干预候选", "red", "L1-05"),
    "goalkeeper_intervention_candidate": ("门将事件", "门将动作候选", "violet", "L1-06"),
}
VERDICTS = {
    "visual_supported_candidate": "视觉证据支持",
    "rejected_candidate": "视觉否定",
    "needs_human_review": "证据不足",
    "conflict_needs_human_review": "动作类型存在分歧",
    "out_of_play_supported_restart_unconfirmed": "出界有证据，重启类型未确认",
    "not_reviewed": "尚未视觉复核",
}
KEEPER_LABELS = {"save": "门将扑救", "catch": "门将接球", "parry": "门将挡球", "punch": "门将拳击球",
                 "foot_pass": "门将脚下出球", "clearance": "门将解围", "rush": "门将出击"}


def cross_check_goalkeeper(result: dict, frames: list[dict], api_key: str) -> None:
    event = result["candidate"]
    indices = [index for index, frame in enumerate(frames) if abs(frame["time_sec"] - event["time_sec"]) <= 1.55]
    focused = [frames[index] for index in indices]
    payload = build_request(event, focused)
    payload["messages"][0]["content"][0]["text"] += (
        "\n本次为独立的门将动作细分复审，没有预设扑救结论。请先辨别疑似门将是否始终站立，"
        "是否有明显支撑腿和摆腿踢球；若站立伸腿将球踢走，应分类foot_pass或clearance，不能因双臂张开就称为侧扑。"
        "区分队友回传后的脚下处理与对方射门后的扑救。不要将禁区中普通防守者的动作套给球门前门将。"
        "请在goalkeeper中额外给出body_state(upright/grounded/airborne/unknown)和foot_kick_visible(true/false)。"
    )
    cross, usage = call_vision_api(payload, api_key)
    cross = validate_review(cross, len(focused))
    cross["evidence_frame_indices"] = [indices[index] for index in cross["evidence_frame_indices"]]
    if cross.get("ball_out_of_play"):
        cross["ball_out_of_play"]["evidence_frame_indices"] = [indices[index] for index in cross["ball_out_of_play"].get("evidence_frame_indices", [])]
    initial = result["vision_review"]
    initial_action = (initial.get("goalkeeper") or {}).get("action_type", "unknown")
    cross_action = (cross.get("goalkeeper") or {}).get("action_type", "unknown")
    agrees = initial["decision"] == cross["decision"] and initial_action == cross_action
    result.update(initial_vision_review=initial, cross_review=cross, cross_review_version=1,
                  cross_review_usage=usage, cross_review_source_indices=indices, vision_review=cross)
    result["fusion_status"] = fusion_status(event, cross) if agrees else "conflict_needs_human_review"
    if not agrees:
        cross["limitations_zh"] = f"初审动作：{initial_action}；独立复审动作：{cross_action}。两次判断不一致，须人工确认。" + cross.get("limitations_zh", "")


def select_candidates(report: dict, maximum: int = 20) -> list[dict]:
    all_events = [event for output in report.get("analyzer_outputs", []) for event in output.get("events", [])]
    quotas = [("goalkeeper_intervention_candidate", 6), ("ball_out_of_play_candidate", 4),
              ("set_piece_delivery_candidate", 3), ("shot_candidate", 4), ("pass_candidate", 4),
              ("defensive_intervention_candidate", 2)]
    selected = []
    for event_type, count in quotas:
        candidates = sorted((event for event in all_events if event["event_type"] == event_type),
                            key=lambda event: event["time_sec"])
        if len(candidates) > count:
            candidates = [candidates[round(i * (len(candidates) - 1) / (count - 1))] for i in range(count)]
        selected.extend(candidates[:max(0, maximum - len(selected))])
    return sorted(selected, key=lambda event: event["time_sec"])


def review_candidates(report: dict, report_path: Path, video: Path, output: Path,
                      selected: list[dict], api_key: str, frame_count: int) -> None:
    gsr = Path(report["source"]["gsr_json"])
    predictions = json.loads(gsr.read_text(encoding="utf-8")).get("predictions", [])
    for event in selected:
        event_id = event["event_id"]
        signature = hashlib.sha256(json.dumps({"event": event, "frames": frame_count, "window": [5, 5],
                                              "schema": 2, "prompt_version": REVIEW_PROMPT_VERSION, "video_size": video.stat().st_size,
                                              "video_mtime": video.stat().st_mtime_ns}, sort_keys=True).encode()).hexdigest()
        review_path = output / "reviews" / f"{event_id}.deepseek.json"
        cached = None
        if review_path.exists():
            previous = json.loads(review_path.read_text(encoding="utf-8"))
            if previous.get("signature") == signature and (previous.get("status") == "reviewed"
                                                          or (not api_key and previous.get("status") == "evidence_ready")):
                if api_key and event["event_type"] == "goalkeeper_intervention_candidate" and not previous.get("cross_review_version"):
                    cached = previous
                else:
                    continue
        focus = {}
        if event["event_type"] == "goalkeeper_intervention_candidate":
            focus = {parse_frame(item["image_id"]): item["bbox_image"] for item in predictions
                     if item.get("track_id") == event.get("actor_track_id") and item.get("bbox_image")}
        print(f"Reviewing {event_id} at {event['time_sec']:.2f}s", flush=True)
        frames = extract_frames(video, event, frame_count, 1280, 5, 5, focus)
        if cached:
            try:
                cross_check_goalkeeper(cached, frames, api_key)
                review_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  keeper cross-check: {cached['fusion_status']}", flush=True)
            except (RuntimeError, ValueError) as exc:
                cached["cross_review_error"] = str(exc)
                cached["fusion_status"] = "needs_human_review"
                review_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        evidence = save_evidence(video, event, frames, output / "evidence" / event_id)
        result = {"schema_version": "2.0", "prompt_version": REVIEW_PROMPT_VERSION, "signature": signature, "candidate": event,
                  "source_report": str(report_path.resolve()), "source_video": str(video.resolve()),
                  "frame_count": len(frames), "focus_frame_count": sum(bool(frame.get("focus_jpeg")) for frame in frames),
                  "evidence_times_sec": [frame["time_sec"] for frame in frames],
                  "evidence_window_sec": [evidence["start_sec"], evidence["end_sec"]],
                  "evidence": evidence, "model": "deepseek-flash"}
        try:
            if not api_key:
                result["status"] = "evidence_ready"
                review_path.parent.mkdir(parents=True, exist_ok=True)
                review_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            review, usage = call_vision_api(build_request(event, frames), api_key)
            review = validate_review(review, len(frames))
            contact = review.get("contact_time_sec")
            if contact is not None and (not isinstance(contact, (int, float))
                                       or not evidence["start_sec"] <= contact <= evidence["end_sec"]):
                raise ValueError("模型动作时间超出审核窗口")
            result.update(vision_review=review, usage=usage, status="reviewed", fusion_status=fusion_status(event, review))
            if event["event_type"] == "goalkeeper_intervention_candidate":
                cross_check_goalkeeper(result, frames, api_key)
        except (RuntimeError, ValueError) as exc:
            result.update(status="failed", error=str(exc))
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {result.get('fusion_status', result['status'])}", flush=True)


def build_bundle(report: dict, selected: list[dict], reviews: dict[str, dict]) -> dict:
    outputs = {item["analyzer"]: item for item in report.get("analyzer_outputs", [])}
    all_events = [event for output in outputs.values() for event in output.get("events", [])]
    events = []
    for event in selected:
        review = reviews.get(event["event_id"], {})
        visual = review.get("vision_review", {}) if review.get("status") == "reviewed" else {}
        status = review.get("fusion_status", "not_reviewed") if visual else "not_reviewed"
        category, label, color, group = PRESENTATION[event["event_type"]]
        subtype = event.get("event_subtype") or event.get("metrics", {}).get("set_piece_type") or "unknown"
        supported = status == "visual_supported_candidate"
        if category == "定位球" and not supported:
            subtype, label = "unknown", "定位球类型待复核"
        if category == "门将事件" and supported:
            label = KEEPER_LABELS.get((visual.get("goalkeeper") or {}).get("action_type"), "门将动作待分类")
        elif category == "门将事件":
            label = "门将动作待复核"
        evidence = review.get("evidence", {})
        start, end = evidence.get("start_sec", max(0, event["time_sec"] - 5)), evidence.get("end_sec", event["time_sec"] + 5)
        images = evidence.get("image_paths", [])
        times = review.get("evidence_times_sec", [])
        events.append({
            "id": event["event_id"], "time": event["time_sec"], "end": min(report["source"]["duration_sec"], end),
            "evidenceStart": start, "category": category, "group": group, "type": label, "title": label,
            "subtype": subtype, "color": color, "reviewStatus": "candidate", "rawEventType": event["event_type"],
            "team": event.get("team"), "actorTrackId": event.get("actor_track_id"),
            "summary": visual.get("short_reason_zh", "尚未获得视觉复核结果"),
            "detail": VERDICTS[status], "visionStatus": status, "visionStatusLabel": VERDICTS[status],
            "confidence": round(float(visual.get("confidence", 0)) * 100),
            "observation": visual.get("observation_zh", ""), "analysis": visual.get("tactical_analysis_zh", ""),
            "limitation": visual.get("limitations_zh", visual.get("uncertainty_reason") or ""),
            "model": review.get("model") if visual else None,
            "frameCount": review.get("frame_count", 0), "focusFrameCount": review.get("focus_frame_count", 0),
            "clipPath": evidence.get("clip_path"),
            "thumbnailPath": images[len(images) // 2] if images else None,
            "evidenceFrames": [{"time": time, "path": file_path} for time, file_path in zip(times, images)],
        })
        if status == "conflict_needs_human_review":
            events[-1].update(
                summary="模型判断存在分歧，暂不采纳具体动作结论。",
                observation="算法在此处发现球员与足球接近的候选，需要回看连续画面确认实际动作。",
                analysis="暂不将此候选计为扑救、接球或脚下出球，也不据此评价动作规范性。"
                if category == "门将事件" else "暂不基于此候选生成确定的技战术结论。",
                modelOpinions=[{"stage": stage, "observation": opinion.get("observation_zh", ""),
                                "analysis": opinion.get("tactical_analysis_zh", "")}
                               for stage, opinion in [("初审", review.get("initial_vision_review", {})),
                                                      ("独立复审", visual)] if opinion],
            )
    counts = []
    for category in dict.fromkeys(value[0] for value in PRESENTATION.values()):
        raw = [event for event in all_events if PRESENTATION.get(event["event_type"], [None])[0] == category]
        subset = [event for event in events if event["category"] == category]
        counts.append({"label": category, "candidates": len(raw), "reviewed": sum(event["visionStatus"] != "not_reviewed" for event in subset),
                       "supported": sum(event["visionStatus"] == "visual_supported_candidate" for event in subset),
                       "rejected": sum(event["visionStatus"] == "rejected_candidate" for event in subset)})
    chains = []
    for event in outputs.get("attacking_chains", {}).get("events", []):
        metrics = event.get("metrics", {})
        shot_review = next((item for item in events if item["rawEventType"] == "shot_candidate"
                            and abs(item["time"] - event["time_sec"]) < 0.3), None)
        if shot_review and shot_review["visionStatus"] in {"rejected_candidate", "conflict_needs_human_review"}:
            continue
        if metrics.get("action_count", 0) < 2:
            continue
        chains.append({"id": event["event_id"], "time": metrics.get("start_sec", event["time_sec"]),
                       "end": event["time_sec"], "title": "可见进攻动作序列",
                       "summary": f"{metrics.get('duration_sec', 0):.1f}秒内有{metrics.get('pass_count', 0)}次传球候选、"
                                  f"{metrics.get('carry_count', 0)}次推进候选。终点射门及整条链路仍需逐项复核。",
                       "actions": metrics.get("actions", []), "status": "候选序列"})
    spatial = []
    for name in ("formation_tendency", "team_shape_engine", "defensive_gaps", "numerical_superiority"):
        for finding in outputs.get(name, {}).get("findings", [])[:3]:
            evidence = finding.get("evidence") or []
            time = evidence[0].get("time_sec") if evidence else finding.get("start_sec")
            if time is None:
                continue
            spatial.append({"id": finding["finding_id"], "time": time, "end": finding.get("end_sec") or time + 5,
                            "title": finding["title_zh"], "summary": finding["summary_zh"],
                            "status": "可见区域几何候选", "limitations": finding.get("limitations_zh", [])})
    return {
        "schema_version": "2.0", "source": report["source"], "events": events,
        "reviewSummary": dict(Counter(event["visionStatus"] for event in events)),
        "layers": {
            "L2": {"title": "事件统计与质量", "counts": counts,
                   "note": "全片算法候选与抽样视觉审核分开计数。视觉支持不等于人工确认。",
                   "unavailable": ["传球成功率：未覆盖全部成功与失败传球", "扑救成功率：缺少完整射正与扑救结果分母", "正式xG：尚无经结果标签校准的模型"]},
            "L3": {"title": "有球组织与进攻序列", "items": chains,
                   "note": "从现有球权和动作时间链提取；视觉否定的射门不会保留为射门进攻链。切镜可能截断序列。"},
            "L4": {"title": "局部空间与队形", "items": spatial,
                   "note": "仅反映转播可见区域及已有场地标定；未出镜球员未知，不代表完整22人阵型。"},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-api", action="store_true")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--frames", type=int, default=41)
    args = parser.parse_args()
    if not 4 <= args.frames <= 61 or not 1 <= args.max_events <= 40:
        parser.error("frames must be 4-61 and max-events 1-40")
    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    if Path(report["source"]["video"]).resolve() != args.video.resolve():
        parser.error("report and source video do not match")
    selected = select_candidates(report, args.max_events)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("DEEPSEEK_API_KEY") if args.review_api else ""
    if args.review_api and not key:
        parser.error("DEEPSEEK_API_KEY is required for --review-api")
    review_candidates(report, args.report_json, args.video, args.output_dir, selected, key, args.frames)
    reviews = {}
    for file_path in (args.output_dir / "reviews").glob("*.deepseek.json"):
        item = json.loads(file_path.read_text(encoding="utf-8"))
        if Path(item["source_video"]).resolve() == args.video.resolve():
            reviews[item["candidate"]["event_id"]] = item
    bundle = build_bundle(report, selected, reviews)
    destination = args.output_dir / "layered_analysis.json"
    destination.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {destination}; review states: {bundle['reviewSummary']}", flush=True)


if __name__ == "__main__":
    main()
