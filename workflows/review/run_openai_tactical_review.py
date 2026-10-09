"""Run bounded OpenAI visual review and publish the existing local web bundle format."""

import argparse
import copy
from collections import Counter
import getpass
import hashlib
import json
import math
import os
from pathlib import Path

from workflows.review.build_layered_tactical_review import PRESENTATION as CANDIDATE_LABELS, build_bundle, select_candidates
from workflows.review.review_tactical_candidates_with_vision import extract_frames, save_evidence
from tactical_analysis.gsr_io import load_gsr_context
from tactical_analysis.openai_review import (MODEL, VERSION, PROMPT, SCHEMA, BudgetExceeded, ReviewError,
    ResponsesReviewer, fingerprint, request_payload, validate_review, write_json)
from tactical_analysis.reviewed_layers import organization_candidates, reviewed_statistics, spatial_coverage


PRESENTATION = {
    "shot": ("射门", "orange", "L1-01"), "set_piece": ("定位球", "yellow", "L1-02"),
    "ball_out": ("出界", "yellow", "L1-02"), "pass": ("传接带", "blue", "L1-03"),
    "cross": ("传接带", "blue", "L1-03"), "carry": ("传接带", "green", "L1-03"),
    "possession_change": ("球权转换", "cyan", "L1-04"),
    "defensive_intervention": ("防守干预", "red", "L1-05"), "goalkeeper": ("门将事件", "violet", "L1-06"),
    "unknown": ("待核实", "blue", "L1-01"),
}
LABELS = {"supported": "画面支持", "corrected": "已修正", "rejected": "未采纳", "uncertain": "待核实"}
EVENT_LABELS = {"shot": "射门", "pass": "传球", "cross": "传中", "carry": "持球推进", "corner": "角球",
    "free_kick": "任意球", "goal_kick": "球门球", "throw_in": "界外球", "kickoff": "中圈开球", "penalty": "点球",
    "touchline": "边线出界", "goal_line": "底线出界", "foot_pass": "脚下出球", "foot_clearance": "大脚出球",
    "save": "扑救", "catch": "接球", "parry": "挡球", "punch": "拳击球", "rush": "出击",
    "aerial_clearance": "争顶解围", "interception": "拦截", "back_pass": "回传", "recycle": "回传组织",
    "switch_pass": "转移传球", "possession_change": "球权转换", "organization": "进攻组织", "spatial": "队形空间", "unknown": "待核实"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_frames(video, start, end, count=21, crop=None):
    center = (start + end) / 2
    frames = extract_frames(video, {"time_sec": center}, count, 1280, center - start, end - center)
    if crop:
        import cv2
        import numpy as np
        x, y, w, h = crop
        for frame in frames:
            image = cv2.imdecode(np.frombuffer(frame["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            region = image[int(y * height):int((y + h) * height), int(x * width):int((x + w) * width)]
            if region.size:
                ok, encoded = cv2.imencode(".jpg", region)
                if ok:
                    frame["focus_jpeg"] = encoded.tobytes()
    return frames


def supplementation(request, task, duration):
    kind = request["kind"]
    if kind == "none":
        return None
    start, end = request["start_sec"], request["end_sec"]
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (start, end)):
        raise ReviewError("Invalid additional evidence window")
    # Bound tool requests to the event neighborhood; the model cannot read arbitrary files.
    start = max(0, task["time"] - 20, start)
    end = min(duration - .05, task.get("end", task["time"]) + 30, end, start + 40)
    if end <= start:
        raise ReviewError("Empty additional evidence window")
    crop = request["crop_xywh"] if kind == "crop" else None
    if crop is not None:
        if len(crop) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in crop):
            raise ReviewError("Invalid requested crop")
        x, y, w, h = crop
        if w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
            raise ReviewError("Crop outside image")
    return start, end, crop


def unresolved(task, start, end, reason):
    return {"decision": "uncertain", "title": "事件待复核", "event_type": "unknown", "subtype": "unknown",
        "team": "unknown", "start_sec": max(start, min(end, task["time"])), "end_sec": end,
        "contact_time_sec": None, "evidence_indices": [], "observation": "尚无足够视觉证据。",
        "interpretation": "暂不输出确定的战术结论。", "advice": "回看连续画面并补充审核。", "limitation": reason,
        "needs_more_evidence": True, "scene_cut": True}


def review_task(task, video, duration, directory, reviewer, teams, signature):
    folder = directory / "evidence" / fingerprint(task)[:16]
    cache = directory / "reviews" / (fingerprint(task)[:16] + ".json")
    key = fingerprint({"task": task, "source": signature, "teams": teams, "model": MODEL,
                       "version": VERSION, "prompt": PROMPT, "schema": SCHEMA})
    if reviewer.model != MODEL or not reviewer.official or reviewer.reasoning_effort != "high" or reviewer.response_format != "schema":
        key = fingerprint({"original": key, "model": reviewer.model, "endpoint": reviewer.api,
                           "reasoning_effort": reviewer.reasoning_effort, "response_format": reviewer.response_format})
    if cache.exists():
        record = json.loads(cache.read_text(encoding="utf-8"))
        if record.get("signature") == key and record.get("status") == "reviewed" and all(
                Path(f["path"]).exists() for f in record["frames"]):
            is_keeper = record["review"]["event_type"] == "goalkeeper" or task.get("candidate_type") == "goalkeeper_intervention_candidate"
            if not is_keeper or record.get("keeper_policy") == "identity-actions-v2":
                return record
    if getattr(reviewer, "cache_only", False):
        raise ReviewError(f"Missing valid review cache for {task['id']}; existing report was not replaced")
    start = max(0, task["time"] - (15 if task["kind"] == "boundary_transition" else 5))
    end = min(duration - .05, task.get("end", task["time"]) + (15 if task["kind"] == "boundary_transition" else 5))
    if task["kind"] == "scan":
        start, end = task["time"], task["end"]
    frames = window_frames(video, start, end)
    calls, history, inspected = [], [], {}
    review = None
    status = "reviewed"
    try:
        for iteration in range(3):
            payload = request_payload(task, frames, teams, previous=review)
            raw, usage = reviewer.call(payload)
            for frame in frames:
                inspected[frame["frame"]] = frame
            calls.append(usage["response_id"])
            review = validate_review(raw, frames)
            history.append({"raw_review": copy.deepcopy(raw), "review": copy.deepcopy(review), "times": [f["time_sec"] for f in frames], "response_id": usage["response_id"]})
            if not review["needs_more_evidence"] or iteration == 2:
                break
            addition = supplementation(review["request"], task, duration)
            if addition is None:
                break
            a, b, crop = addition
            extra = window_frames(video, a, b, crop=crop)
            # Preserve original temporal context, while adding denser or wider evidence.
            combined = {f["frame"]: f for f in frames}
            combined.update({f["frame"]: f for f in extra})
            frames = sorted(combined.values(), key=lambda f: f["time_sec"])
            if len(frames) > 63:
                frames = frames[::2]
        if review["event_type"] == "goalkeeper" and review["decision"] in {"supported", "corrected"}:
            raw, usage = reviewer.call(request_payload(task, frames, teams, independent=True))
            calls.append(usage["response_id"])
            other = validate_review(raw, frames)
            history.append({"raw_review": copy.deepcopy(raw), "independent_keeper_review": other, "response_id": usage["response_id"]})
            if (other["decision"] not in {"supported", "corrected"} or
                    any(review[k] != other[k] for k in ("event_type", "subtype", "team"))):
                review.update(decision="uncertain", interpretation="两次动作判断不一致，暂不计为确定门将事件。",
                              limitation="门将独立复审存在分歧；" + review["limitation"])
    except (ReviewError, ValueError) as exc:
        status = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "failed"
        review = unresolved(task, start, end, str(exc))
    prepared = {f["frame"]: f for f in frames}
    prepared.update(inspected)
    all_frames = sorted(prepared.values(), key=lambda f: f["time_sec"])
    first, last = all_frames[0]["time_sec"], all_frames[-1]["time_sec"]
    evidence = save_evidence(video, {"time_sec": (first + last) / 2}, all_frames, folder,
                             (last - first) / 2, (last - first) / 2)
    paths = [{"time": f["time_sec"], "path": p} for f, p in zip(all_frames, evidence["image_paths"])]
    checked = [p for f, p in zip(all_frames, paths) if f["frame"] in inspected]
    record = {"signature": key, "task": task, "status": status, "review": review,
              "history": history, "response_ids": calls, "frames": paths, "inspected_frames": checked,
              "evidence": evidence, "model": reviewer.model, "endpoint": reviewer.api,
              "keeper_policy": "identity-actions-v2"}
    write_json(cache, record)
    print(f"{task['id']} {task['time']:.2f}s: {status}/{review['decision']} {review['subtype']}", flush=True)
    return record


def event_from_record(record):
    r, task, evidence = record["review"], record["task"], record["evidence"]
    category, color, group = PRESENTATION.get(r["event_type"], PRESENTATION["unknown"])
    subtype = "pass" if category == "传接带" and r["subtype"] == "foot_pass" else r["subtype"]
    accepted = record["status"] == "reviewed" and r["decision"] in {"supported", "corrected"}
    verdict = "visual_supported_candidate" if accepted else ("rejected_candidate" if r["decision"] == "rejected" else "needs_human_review")
    timestamp = r["contact_time_sec"] if r["contact_time_sec"] is not None else r["start_sec"]
    return {"id": task["id"], "time": timestamp, "end": r["end_sec"], "title": r["title"], "type": EVENT_LABELS[subtype],
        "category": category, "group": group, "color": color, "subtype": subtype,
        "rawEventType": task.get("candidate_type", "scan"), "team": None if r["team"] == "unknown" else r["team"],
        "actorTrackId": None, "sceneId": task["id"] if r["scene_cut"] else None,
        "reviewStatus": "rejected" if r["decision"] == "rejected" else "candidate", "reviewDecision": r["decision"],
        "visionStatus": verdict, "visionStatusLabel": LABELS[r["decision"]], "detail": LABELS[r["decision"]],
        "publishedForStatistics": accepted, "summary": r["observation"], "observation": r["observation"],
        "analysis": r["interpretation"], "limitation": r["limitation"],
        "commentary": {k: r[k] for k in ("observation", "interpretation", "advice", "limitation")},
        "reviewerLabel": "智能复核", "model": None, "reviewModel": record.get("model", MODEL), "reviewedFrameCount": len(record["inspected_frames"]),
        "frameCount": len(record["frames"]), "evidenceFrames": record["frames"], "inspectedFrames": record["inspected_frames"],
        "evidenceStart": evidence["start_sec"], "evidenceEnd": evidence["end_sec"],
        "clipPath": evidence["clip_path"], "thumbnailPath": record["frames"][len(record["frames"]) // 2]["path"],
        "originalCandidate": {"title": CANDIDATE_LABELS.get(task.get("candidate_type"), ("", "分段扫描"))[1], "time": task["time"]}}


def deduplicate(events):
    result = []
    for event in sorted(events, key=lambda e: e["time"]):
        duplicate = next((e for e in result if e["publishedForStatistics"] and event["publishedForStatistics"]
                          and ((e.get("category") == event.get("category") == "出界")
                               or (e["subtype"] == event["subtype"] and e["team"] == event["team"]))
                          and abs(e["time"] - event["time"]) <= 1.5), None)
        if duplicate:
            if duplicate["id"].startswith("scan-") and not event["id"].startswith("scan-"):
                result[result.index(duplicate)] = {**duplicate, "publishedForStatistics": False,
                    "duplicateOf": event["id"], "visionStatusLabel": "重复证据，不重复统计"}
                result.append(event)
                continue
            event = {**event, "publishedForStatistics": False, "duplicateOf": duplicate["id"],
                     "visionStatusLabel": "重复证据，不重复统计"}
            if event.get("category") == "出界" and event["team"] != duplicate["team"]:
                index = result.index(duplicate)
                result[index] = {**duplicate, "team": None, "teamConflict": True}
        result.append(event)
    return result


def compare_baseline(events, baseline):
    by_id = {e["id"]: e for e in events}
    rows = []
    for previous in baseline["events"]:
        current = by_id.get(previous["id"])
        rows.append({"id": previous["id"], "previous_subtype": previous.get("subtype"),
            "current_subtype": current.get("subtype") if current else None,
            "same_subtype": current is not None and current.get("subtype") == previous.get("subtype"),
            "same_team": current is not None and current.get("team") == previous.get("team"),
            "time_delta_sec": round(current["time"] - previous["time"], 3) if current else None})
    return {"note": "仅与旧助手版本对照，不是人工真值准确率；审核输入不含旧审核结论。", "rows": rows}


def audit_boundaries(records, video, duration, directory, reviewer, teams, signature):
    updated = []
    for record in records:
        r = record["review"]
        if r["event_type"] == "ball_out" and r["decision"] in {"supported", "corrected"}:
            task = {"id": "boundary-" + record["task"]["id"], "kind": "boundary_transition",
                    "time": record["task"]["time"], "candidate_type": "new_boundary_crossing"}
            audit = review_task(task, video, duration, directory, reviewer, teams, signature)
            refined = dict(audit["review"])
            if (refined["event_type"] == "ball_out" and refined["contact_time_sec"] is None
                    and refined["decision"] in {"supported", "corrected"}):
                refined.update(decision="uncertain", interpretation="未定位到新的越界过程，不能将死球等待重复计为出界。",
                               limitation="缺少新越界时间；" + refined["limitation"])
            record = {**audit, "task": record["task"], "review": refined,
                      "boundary_audit": True, "initial_record_signature": record["signature"]}
        updated.append(record)
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bundle", type=Path)
    inputs.add_argument("--report-json", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--prompt-key", action="store_true")
    parser.add_argument("--cached-only", action="store_true", help="Re-publish existing reviews without any network calls")
    parser.add_argument("--max-calls", type=int, default=80)
    parser.add_argument("--max-usd", type=float, default=8)
    parser.add_argument("--no-cost-limit", action="store_true", help="No monetary cap; token usage is still recorded")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--response-format", choices=["schema", "json_prompt"], default="schema")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-events", type=int, default=24)
    parser.add_argument("--scan-seconds", type=int, default=30)
    parser.add_argument("--team-left", default="蓝衣，黄绿色门将")
    parser.add_argument("--team-right", default="白衣")
    args = parser.parse_args()
    if args.max_calls < 1 or not 0 < args.max_usd <= 100 or not 1 <= args.max_events <= 120 or args.scan_seconds < 10:
        parser.error("Invalid bounded review settings")
    from tactical_analysis.openai_review import responses_endpoint, API
    endpoint = responses_endpoint(args.base_url)
    if endpoint != API and args.api_key_env == "OPENAI_API_KEY" and not (args.prompt_key or args.cached_only):
        parser.error("Use a separate --api-key-env RELAY_API_KEY or --prompt-key for a third-party endpoint")
    key = getpass.getpass("API key (hidden): ").strip() if args.prompt_key else os.environ.get(args.api_key_env, "")
    if not key and not args.cached_only:
        parser.error("Set the selected backend key environment variable or use --prompt-key")
    source_path = args.bundle or args.report_json
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if args.report_json:
        source = build_bundle(source, select_candidates(source, args.max_events), {})
    video = (args.video or Path(source["source"]["video"])).resolve()
    if video != Path(source["source"]["video"]).resolve():
        parser.error("Source video differs from report")
    duration = float(source["source"]["duration_sec"])
    import cv2
    capture = cv2.VideoCapture(str(video))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    actual_duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / max(actual_fps, 1e-6)
    capture.release()
    if duration <= .1 or abs(actual_duration - duration) > .2 or abs(actual_fps - source["source"]["fps"]) > .1:
        parser.error("Video metadata differs from the source report")
    context = load_gsr_context(Path(source["source"]["gsr_json"]), source["source"]["fps"])
    teams = {"left": args.team_left, "right": args.team_right, "referee": "裁判独立于两队，不按候选ID猜测身份"}
    signature = {"video_sha256": file_hash(video), "source_sha256": file_hash(source_path)}
    reviewer = ResponsesReviewer(key, args.output_dir, args.max_calls,
        None if args.no_cost_limit else args.max_usd, model=args.model, base_url=args.base_url,
        reasoning_effort=args.reasoning_effort, response_format=args.response_format)
    reviewer.cache_only = args.cached_only
    tasks = [{"id": e["id"], "kind": "event", "time": e["time"], "candidate_type": e.get("rawEventType", "unknown")}
             for e in source["events"][:args.max_events]]
    tasks += [{"id": f"scan-{t:05d}", "kind": "scan", "time": float(t), "end": min(duration - .05, t + args.scan_seconds)}
              for t in range(0, int(duration), args.scan_seconds)]
    records = []
    for task in tasks:
        records.append(review_task(task, video, duration, args.output_dir, reviewer, teams, signature))
    records = audit_boundaries(records, video, duration, args.output_dir, reviewer, teams, signature)
    events = deduplicate([event_from_record(r) for r in records if r["task"]["kind"] != "scan"
                          or (r["status"] == "reviewed" and r["review"]["decision"] in {"supported", "corrected"})])
    higher = {"L3": [], "L4": []}
    higher_tasks = [("L3", {**r, "kind": "organization"}) for r in organization_candidates(events)[:8]]
    higher_tasks += [("L4", {"id": item["id"], "time": item["time"], "end": min(duration - .05, item["time"] + 5),
                            "kind": "spatial"}) for item in source["layers"].get("L4", {}).get("items", [])[:8]]
    for level, task in higher_tasks:
        record = review_task(task, video, duration, args.output_dir, reviewer, teams, signature)
        records.append(record)
        event = event_from_record(record)
        r = record["review"]
        item = {**event, "decision": r["decision"], "status": LABELS[r["decision"]],
                "sourceEventIds": task.get("sourceEventIds", []), "numericConclusionPublished": False}
        if level == "L4":
            item["quality"] = spatial_coverage(context, task["time"])
        higher[level].append(item)
    categories = list(dict.fromkeys(v[0] for v in PRESENTATION.values()))
    counts = [{"label": c, "candidates": sum(e["category"] == c for e in events),
               "reviewed": sum(e["category"] == c and e["reviewedFrameCount"] > 0 for e in events),
               "supported": sum(e["category"] == c and e["publishedForStatistics"] for e in events),
               "rejected": sum(e["category"] == c and e["reviewDecision"] == "rejected" for e in events)} for c in categories]
    bundle = {"schema_version": "3.0", "source": source["source"], "events": events,
        "reviewSummary": dict(Counter(e["visionStatus"] for e in events)),
        "layers": {"L2": {"title": "数据统计", "counts": counts,
            "note": "仅统计本次复核采纳的节点；分段抽帧扫描不是逐帧全量检测，不代表全场事件总数。",
            "reviewedMetrics": reviewed_statistics(events),
            "unavailable": ["正式xG与扑救成功率：缺少结果标签和完整分母", "精确阵型与空间距离：缺少完整可见性及标定误差验证"]},
            **{level: {"title": "进攻组织" if level == "L3" else "队形空间", "items": higher[level],
               "note": "连续抽帧复核；仅描述可见区域，观察与解释分开呈现。"} for level in higher}},
        "reviewMetadata": {"mode": "openai_api" if reviewer.official else "compatible_api",
            "model": reviewer.model, "endpoint": reviewer.api, "workflow_version": VERSION,
            "reasoning_effort": reviewer.reasoning_effort,
            "response_format": reviewer.response_format,
            "provider_identity_verified": reviewer.official, "monetary_cap_usd": reviewer.max_usd,
            "boundary_transition_audit": True,
            "human_ground_truth": False, **signature, "api_calls": len(reviewer.ledger),
            "api_calls_this_run": reviewer.calls_this_run,
            "estimated_cost_upper_usd": round(sum(r["cost_upper_usd"] for r in reviewer.ledger), 4) if reviewer.priced else None,
            "task_states": dict(Counter(r["status"] for r in records)), "team_context": teams},
        "presentation": {"title": f"{video.stem} 智能视觉复盘", "providerNamesVisible": False}}
    write_json(args.output_dir / "layered_analysis.json", bundle)
    if args.baseline:
        write_json(args.output_dir / "baseline_comparison.json", compare_baseline(events, json.loads(args.baseline.read_text(encoding="utf-8"))))
    lines = ["# 智能视觉复盘", "", "抽帧复核结果，非人工专家真值；不保证与旧版本同等准确。", "",
             f"调用记录：{len(reviewer.ledger)}；费用：" + (f"估算 ${bundle['reviewMetadata']['estimated_cost_upper_usd']}" if reviewer.priced else "中转站价格未知，以平台账单为准，本次不设金额上限") + "。", ""]
    for event in events + higher["L3"] + higher["L4"]:
        lines += [f"## {event['time']:.2f}s {event['title']}（{event['visionStatusLabel']}）", ""]
        for k, label in (("observation", "观察"), ("interpretation", "解读"), ("advice", "建议"), ("limitation", "边界")):
            lines += [f"{label}：{event['commentary'][k]}", ""]
    (args.output_dir / "tactical_review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(bundle["reviewMetadata"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
