"""Publish reviewed L3/L4 analysis and evidence-linked commentary for the local UI."""

import argparse
import copy
import json
from pathlib import Path

from workflows.review.build_assistant_review_version import digest
from tactical_analysis.gsr_io import load_gsr_context
from tactical_analysis.reviewed_layers import organization_candidates, reviewed_statistics, spatial_coverage


LABELS = {"supported": "画面支持", "corrected": "已修正", "rejected": "未采纳", "uncertain": "待核实"}
ADVICE = {
    "cross": "结合传中时的前点、后点跑动和外围接应复盘，不只看球的终点。",
    "pass": "观察接球前的朝向与邻近支援，区分完成传递和真正获得向前空间。",
    "carry": "比较继续持球和提前分球的选择，关注对手收缩前的出球时机。",
    "aerial_clearance": "解围后继续观察第二落点与队友接应，不将第一点触球当作稳定夺回球权。",
    "foot_clearance": "观察长球目标区的队友接应和争顶后保护；不以一次大脚评价扑救表现。",
    "foot_pass": "关注门将接球前的朝向、后卫接应角度和短传线路，而非扑救动作。",
    "throw_in": "将开出前准备与接球过程保留为独立素材，本阶段不推断定位球套路。",
    "recycle": "回传需要结合前方线路是否受阻判断，不能单独据此评价进攻消极。",
    "back_pass": "回看回传前的压迫与后方出口，再判断是否成功保留球权。",
    "switch_pass": "补充接球端画面，确认接收与后续空间后再评价转移效果。",
}


def enrich_bundle(source, annotations, evidence, context):
    result = copy.deepcopy(source)
    event_index = {e["id"]: e for e in result["events"]}
    for event_id, team in annotations.get("teamCorrections", {}).items():
        if event_id not in event_index or team not in {"left", "right", None}:
            raise ValueError("Invalid reviewed team override")
        event_index[event_id]["teamBeforeReview"] = event_index[event_id].get("team")
        event_index[event_id]["team"] = team
    for event in result["events"]:
        event["reviewerLabel"] = "智能复核"
        event["visionStatusLabel"] = LABELS[event["assistantDecision"]]
        event["detail"] = event["visionStatusLabel"]
        event["commentary"] = {
            "observation": event.get("observation", ""), "interpretation": event.get("analysis", ""),
            "advice": ADVICE.get(event["subtype"], "先核实触球、边界或球权变化证据，再纳入战术统计。"),
            "limitation": event.get("limitation", ""), "source": "智能分析参考，非专家评分",
        }
    rules = organization_candidates(result["events"])
    windows = {e["id"]: e for e in evidence["events"]}
    reviews = annotations["reviews"]
    if len({r["id"] for r in reviews}) != len(reviews) or {r["id"] for r in reviews} != set(windows):
        raise ValueError("Each prepared evidence window needs exactly one review")
    layers = {"L3": [], "L4": []}
    for review in reviews:
        window = windows[review["id"]]
        if review["level"] not in layers or review["decision"] not in LABELS:
            raise ValueError("Invalid layer or decision")
        if not (window["evidenceStart"] - 1e-6 <= review["time"] <= review["end"] <= window["evidenceEnd"] + 1e-6):
            raise ValueError("Review interval is outside inspected evidence")
        inspected = window["evidenceFrames"][::2]
        item = {**review, "status": LABELS[review["decision"]], "summary": review["observation"],
                "commentary": {key: review[key] for key in ("observation", "interpretation", "advice", "limitation")},
                "inspectedFrames": inspected, "clipPath": window["clipPath"],
                "evidenceStart": window["evidenceStart"], "evidenceEnd": window["evidenceEnd"],
                "thumbnailPath": inspected[len(inspected) // 2]["path"],
                "sourceCandidate": {key: window[key] for key in ("id", "time", "title", "summary")},
                "publishedForStatistics": review["decision"] in {"supported", "corrected"}}
        if review["level"] == "L3":
            matches = [r for r in rules if r["kind"] == review["ruleKind"] and review["time"] <= r["time"] <= review["end"]]
            if not matches:
                raise ValueError(f"No accepted event rule supports {review['id']}")
            item["sourceEventIds"] = matches[0]["sourceEventIds"]
            item["ruleEvidence"] = matches[0]
        else:
            item["quality"] = spatial_coverage(context, window["time"])
            item["numericConclusionPublished"] = False
        layers[review["level"]].append(item)
    result["layers"]["L2"]["note"] = "仅统计复核节点，不代表全场事件总数；无法据此计算全场成功率。"
    result["layers"]["L2"]["reviewedMetrics"] = reviewed_statistics(result["events"])
    result["layers"]["L3"] = {"title": "有球组织与进攻过程", "items": layers["L3"],
        "note": "同队事件邻接规则筛选，结合时序抽帧复核。观察、战术解读与复盘建议分开呈现。"}
    result["layers"]["L4"] = {"title": "无球站位与局部空间", "items": layers["L4"],
        "note": "仅描述可见区域。未采纳的阵型与人数优势保留审核依据，不计入有效战术结论。"}
    result["reviewMetadata"].update(reviewer=annotations["reviewer"], scope=annotations["scope"], api_calls=0,
                                      human_ground_truth=False, higher_layer_windows=len(reviews),
                                      higher_layer_images=sum(len(w["evidenceFrames"][::2]) for w in windows.values()))
    result["presentation"] = {"title": "五分钟战术深度复盘", "providerNamesVisible": False}
    result["organizationRuleCandidates"] = rules
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.bundle.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    if digest(Path(source["source"]["video"])) != annotations["source_video_sha256"] or digest(args.bundle) != annotations["source_bundle_sha256"]:
        raise ValueError("Source video changed; review again before publishing")
    evidence = json.loads((args.output_dir / "review_evidence.json").read_text(encoding="utf-8"))
    context = load_gsr_context(Path(source["source"]["gsr_json"]), source["source"]["fps"])
    result = enrich_bundle(source, annotations, evidence, context)
    result["reviewMetadata"]["source_bundle_sha256"] = digest(args.bundle)
    (args.output_dir / "layered_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 五分钟战术深度复盘", "", annotations["scope"], "", "主视频复用检测框版本，本轮不调用外部API。", ""]
    for level in ("L3", "L4"):
        lines += [f"## {level} {result['layers'][level]['title']}", ""]
        for item in result["layers"][level]["items"]:
            lines += [f"### {item['time']:.2f}s {item['title']}（{item['status']}）", ""]
            for key, label in (("observation", "观察"), ("interpretation", "解读"), ("advice", "建议"), ("limitation", "限制")):
                lines += [f"{label}：{item[key]}", ""]
    (args.output_dir / "tactical_review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({level: len(result["layers"][level]["items"]) for level in ("L3", "L4")}))


if __name__ == "__main__":
    main()
