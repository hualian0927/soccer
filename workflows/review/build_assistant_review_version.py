"""Publish explicitly authored visual-review annotations without making API calls."""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

from workflows.review.review_tactical_candidates_with_vision import extract_frames, save_evidence


LABELS = {"supported": "助手支持", "corrected": "助手改判", "rejected": "助手排除", "uncertain": "仍待核实"}
GROUPS = {"射门": ("L1-01", "orange"), "定位球": ("L1-02", "yellow"), "出界": ("L1-02", "yellow"),
          "传接带": ("L1-03", "blue"), "防守干预": ("L1-05", "red"), "门将事件": ("L1-06", "violet")}


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def build_review_bundle(source, annotations):
    result = copy.deepcopy(source)
    originals = {event["id"]: event for event in source["events"]}
    records = annotations["reviews"]
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)) or {r["id"] for r in records if not r.get("added")} != set(originals):
        raise ValueError("Annotations must cover every original event exactly once")
    reviewed = []
    for record in records:
        evidence_source = originals[record.get("evidence_source_id", record["id"])]
        original = originals.get(record["id"])
        frames = evidence_source["evidenceFrames"]
        indices = record.get("inspected_indices", sorted(set([*range(0, len(frames), 4), len(frames) - 1,
                                                              *record.get("additional_indices", [])])))
        if not indices or any(index < 0 or index >= len(frames) for index in indices):
            raise ValueError("Invalid inspected frame index")
        if not 0 <= record["time"] < source["source"]["duration_sec"]:
            raise ValueError("Review timestamp exceeds source duration")
        if record["decision"] not in LABELS or record["category"] not in GROUPS:
            raise ValueError("Unknown decision or category")
        event = copy.deepcopy(evidence_source)
        group, color = GROUPS[record["category"]]
        accepted = record["decision"] in {"supported", "corrected"}
        event.update(
            id=record["id"], time=record["time"], end=min(source["source"]["duration_sec"], record["time"] + 5),
            evidenceStart=max(0, record["time"] - 5), category=record["category"], subtype=record["subtype"],
            group=group, color=color, title=record["title"], type=record["title"], summary=record["observation"],
            observation=record["observation"], analysis=record["analysis"], limitation=record["limitation"],
            detail=LABELS[record["decision"]], visionStatusLabel=LABELS[record["decision"]],
            visionStatus="visual_supported_candidate" if accepted else
            "rejected_candidate" if record["decision"] == "rejected" else "needs_human_review",
            reviewStatus="candidate", confidence=None, model=None, focusFrameCount=0,
            reviewerLabel=annotations["reviewer"], assistantDecision=record["decision"],
            publishedForStatistics=accepted, originalCandidate=copy.deepcopy(original),
            inspectedFrames=[frames[index] for index in indices], reviewedFrameCount=len(indices),
            observedInterval=record["evidence_interval"], sourceEventId=evidence_source["id"],
        )
        event.pop("modelOpinions", None)
        if record.get("added"):
            event["rawEventType"] = "assistant_added_restart"
            event["team"] = None
            event["actorTrackId"] = None
        if record.get("team"):
            event["team"] = record["team"]
        reviewed.append(event)
    result["events"] = sorted(reviewed, key=lambda event: event["time"])
    result["reviewSummary"] = dict(Counter(event["assistantDecision"] for event in reviewed))
    result["reviewMetadata"] = {"reviewer": annotations["reviewer"], "scope": annotations["scope"],
                                "method": annotations["method"], "api_calls": 0, "human_ground_truth": False,
                                "source_video_sha256": annotations["source_video_sha256"]}
    counts = []
    for category in GROUPS:
        subset = [event for event in reviewed if event["category"] == category]
        counts.append({"label": category, "candidates": sum(e["category"] == category for e in source["events"]),
                       "reviewed": len(subset), "supported": sum(e["publishedForStatistics"] for e in subset),
                       "rejected": sum(e["assistantDecision"] == "rejected" for e in subset)})
    result["layers"]["L2"].update(title="本次切片复核统计", counts=counts,
        columns=["类别", "原节点", "本版", "采纳", "排除"],
        note="仅统计已查验的17个候选和1个补充节点，非全场事件总数。改判后类别会变化；助手意见不等于人工真值。")
    result["layers"]["L3"] = {"title": "已看片段中的组织过程", "note": "由助手依据本次抽帧复核整理，不沿用已被改判的射门链。仅描述可见过程。", "items": [
        {"id": "review-chain-1", "time": 12.833, "end": 20.833, "title": "白队短传拉边、传中与外围接应",
         "status": "助手片段解读", "summary": "白队转移至右侧后传中；禁区争顶后球回到外沿，白队继续接应。解围第一点不等于蓝队稳定夺回球权。"},
        {"id": "review-chain-2", "time": 140.633, "end": 147.633, "title": "蓝队中路接应后纵向推进",
         "status": "助手片段解读", "summary": "中场出球和多人接应后，蓝衣球员向禁区方向推进，白队回追收缩。不能仅凭该片段给出过人成功率或射门转化率。"},
        {"id": "review-chain-3", "time": 242.567, "end": 258.333, "title": "蓝队利用门将循环组织后场出球",
         "status": "助手片段解读", "summary": "回做门将、短传给后卫、再次回做再向侧方出球，构成连续后场组织。两次门将触球均不应被包装为扑救。"},
    ]}
    result["layers"]["L4"]["note"] = "沿用上一版几何候选，本轮没有独立复核阵型和空间数值；不要将本页当作助手已确认结论。"
    for item in result["layers"]["L4"]["items"]:
        item["status"] = "原算法候选，本轮未复核"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--boxed-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    for file_path, expected in [(args.video, annotations["source_video_sha256"]),
                                (args.source_bundle, annotations["source_bundle_sha256"])]:
        if digest(file_path) != expected:
            raise ValueError(f"Reviewed source has changed: {file_path}")
    if args.output_dir.resolve() == args.source_bundle.parent.resolve():
        raise ValueError("Use a separate version directory")
    source = json.loads(args.source_bundle.read_text(encoding="utf-8"))
    result = build_review_bundle(source, annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for event in result["events"]:
        print(f"Preparing {event['id']} at {event['time']:.3f}s", flush=True)
        frames = extract_frames(args.video, {"time_sec": event["time"]}, 41, 1280, 5, 5)
        evidence = save_evidence(args.video, {"time_sec": event["time"]}, frames, args.output_dir / "evidence" / event["id"])
        event.update(clipPath=evidence["clip_path"], evidenceStart=evidence["start_sec"], end=evidence["end_sec"],
                     frameCount=len(frames), thumbnailPath=evidence["image_paths"][len(frames) // 2],
                     evidenceFrames=[{"path": p, "time": f["time_sec"]} for p, f in zip(evidence["image_paths"], frames)])
    destination = args.output_dir / "test_5min_assistant_review.mp4"
    shutil.copy2(args.boxed_video, destination)
    shutil.copy2(args.annotations, args.output_dir / "assistant_annotations.json")
    (args.output_dir / "layered_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 五分钟助手切片复核版", "", annotations["scope"], "", "本轮未调用DeepSeek；主视频复用上一版原片加框的画面，新版变化是复核结论及按修正时间重新切出的证据。", "",
             "## 复核统计", "", str(result["reviewSummary"]), "", "## 逐节点结果", "",
             "| 时间 | 原候选 | 复核结果 | 处理 |", "| --- | --- | --- | --- |"]
    for e in result["events"]:
        original = e["originalCandidate"]
        lines.append(f"| {e['time']:.3f}s | {original['title'] if original else '切片中新发现'} | {e['title']} | {e['visionStatusLabel']} |")
    for e in result["events"]:
        lines += ["", f"## {e['time']:.3f}s {e['title']}", "", f"观察：{e['observation']}", "",
                  f"解读：{e['analysis']}", "", f"限制：{e['limitation']}", "",
                  f"已查验{e['reviewedFrameCount']}张来源抽帧；新审核片段：{e['evidenceStart']:.3f}–{e['end']:.3f}s。"]
    (args.output_dir / "assistant_review_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["reviewSummary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
