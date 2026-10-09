"""Review a local tactical event candidate with DeepSeek vision frames."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error
import urllib.request


API_URL = "https://api.deepseek.com/chat/completions"
REVIEW_PROMPT_VERSION = "2026-09-20-v3"
SUPPORTED_TYPES = {
    "shot_candidate",
    "corner_candidate",
    "set_piece_delivery_candidate",
    "goalkeeper_intervention_candidate",
    "defensive_intervention_candidate",
    "pass_candidate",
    "ball_out_of_play_candidate",
}
DECISIONS = {"confirmed", "rejected", "uncertain"}
EVENT_NAMES = {
    "shot_candidate": "shot",
    "corner_candidate": "set_piece_delivery",
    "set_piece_delivery_candidate": "set_piece_delivery",
    "goalkeeper_intervention_candidate": "goalkeeper_intervention",
    "defensive_intervention_candidate": "defensive_intervention",
    "pass_candidate": "pass",
    "ball_out_of_play_candidate": "ball_out_of_play",
}
EVENT_ALIASES = {
    "corner": "set_piece_delivery",
    "free_kick": "set_piece_delivery",
    "goal_kick": "set_piece_delivery",
    "throw_in": "set_piece_delivery",
    "kickoff": "set_piece_delivery",
    "penalty": "set_piece_delivery",
}
SET_PIECE_SUBTYPES = {"corner", "free_kick", "goal_kick", "throw_in", "kickoff", "penalty"}
SET_PIECE_ALIASES = {
    "corner_kick": "corner",
    "throw-in": "throw_in",
    "throw_in": "throw_in",
    "free-kick": "free_kick",
    "goal-kick": "goal_kick",
}


def is_set_piece(event: dict) -> bool:
    return event.get("event_type") in {"corner_candidate", "set_piece_delivery_candidate"}


def evidence_bounds(event: dict, duration_sec: float, before_sec: float, after_sec: float) -> tuple[float, float]:
    center = float(event["time_sec"])
    if not all(math.isfinite(value) for value in (center, duration_sec, before_sec, after_sec)):
        raise ValueError("时间参数必须为有限数值")
    if not 0 <= center < duration_sec:
        raise ValueError("候选事件时间不在视频范围内")
    start, end = center - before_sec, center + after_sec
    start = max(0.0, start)
    end = min(duration_sec - 0.05, end)
    if end <= start:
        raise ValueError("候选事件时间不在视频范围内")
    return round(start, 3), round(end, 3)


def find_candidate(report: dict, event_id: str) -> dict:
    matches = [
        event
        for analyzer in report.get("analyzer_outputs", [])
        for event in analyzer.get("events", [])
        if event.get("event_id") == event_id
    ]
    if len(matches) != 1:
        raise ValueError(f"候选事件 ID 应唯一且存在，实际匹配 {len(matches)} 个: {event_id}")
    event = matches[0]
    if event.get("event_type") not in SUPPORTED_TYPES:
        raise ValueError(f"尚未支持复核事件类型: {event.get('event_type')}")
    return event


def sample_times(
    event: dict, duration_sec: float, count: int, before_sec: float = 5.0, after_sec: float = 5.0
) -> list[float]:
    if count < 2:
        raise ValueError("至少需要两张证据帧")
    start, end = evidence_bounds(event, duration_sec, before_sec, after_sec)
    return [round(start + (end - start) * i / (count - 1), 3) for i in range(count)]


def extract_frames(
    video_path: Path, event: dict, count: int, width: int,
    before_sec: float = 5.0, after_sec: float = 5.0,
    focus_boxes: dict[int, dict] | None = None,
) -> list[dict]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法读取视频: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frames <= 0:
            raise ValueError("视频缺少有效帧率或帧数")
        duration = frames / fps
        samples = []
        for second in sample_times(event, duration, count, before_sec, after_sec):
            source_index = min(int(frames) - 1, round(second * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"无法截取 {second:.2f} 秒附近的帧")
            height, original_width = frame.shape[:2]
            focus_jpeg = None
            box = (focus_boxes or {}).get(source_index + 1)
            if box and box.get("w", 0) > 0 and box.get("h", 0) > 0:
                padding = max(50, box["h"] * 0.7)
                x1, y1 = max(0, int(box["x"] - padding)), max(0, int(box["y"] - padding))
                x2 = min(original_width, int(box["x"] + box["w"] + padding))
                y2 = min(height, int(box["y"] + box["h"] + padding))
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    encoded_ok, encoded_crop = cv2.imencode(".jpg", crop)
                    if encoded_ok:
                        focus_jpeg = encoded_crop.tobytes()
            if original_width > width:
                frame = cv2.resize(
                    frame,
                    (width, round(height * width / original_width)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                raise ValueError("JPEG 编码失败")
            samples.append({"time_sec": round(source_index / fps, 3), "jpeg": encoded.tobytes(),
                            "focus_jpeg": focus_jpeg, "frame": source_index + 1})
        return samples
    finally:
        capture.release()


def build_request(event: dict, frames: list[dict]) -> dict:
    context = {
        "candidate_type": "possible_restart" if is_set_piece(event) else event["event_type"],
        "candidate_time_sec": event["time_sec"],
        "team": None if is_set_piece(event) else event.get("team"),
        "rule_confidence": None if is_set_piece(event) else event.get("confidence"),
        "metrics": {
            key: value
            for key, value in event.get("metrics", {}).items()
            if not is_set_piece(event) and key in {"shot_distance_m"}
        },
    }
    instructions = (
        "你是足球比赛事件复核员。按时间顺序查看以下原始视频帧；二维报告只是可能错误的候选，不可当作事实。"
        "请判定候选动作是否真的发生，并区分射门/传中/解围、定位球类型、门将扑救/普通接球等。"
        "定位球候选的几何时间可能只是足球出界，并非重新开球。请跨完整时间窗寻找出界、等待、重启三个阶段；"
        "只有清楚看到角旗区以脚开球才能确认角球，边线附近掷球应标为 throw_in，球门区开球应标为 goal_kick。"
        "不要把角球区附近的普通比赛动作、出界后的等待或远离角旗的边线动作称为角球。"
        "不能仅凭球靠近角旗区或候选标签确认角球；若重启未进入取样画面，返回 uncertain。"
        "静态抽帧不足以证明触球瞬间、球高或最终结果；遮挡、切镜或证据不充分时必须返回 uncertain。"
        "本次证据窗为候选时刻前后各5秒；只描述这个窗口内真正可见的内容。"
        "球整体越过边线或球门线可支持球出界，但坐标越界可能是标定误差；出界不等于已发生角球、任意球或球门球。"
        "若仅看清出界、没有看清重启，分别填写ball_out_of_play.observed=true与restart_observed=false，定位球类型保持null。"
        "门将必须结合服装、球门位置和连续身份确认；倒地或身体水平不等于头球，头球必须看清头部触球。"
        "仅侧向跑动不能称为侧扑；普通接回传、持球发球不能称为扑救；须观察来球、身体动作、球接触和后续结果。"
        "门将局部图是同一时刻的辅助裁剪，可能跟错人，应以完整画面核实身份。"
        "仅凭无 ID 标注的原始帧不要猜测 actor_track_id，也不要凭画面估计精确球速、距离、xG。"
        "不要猜测国家、俱乐部、运动员姓名或把比分字幕缩写映射到球衣；只使用画面中的球衣颜色、持球方、防守方称呼。"
        "仅返回一个 JSON 对象，字段为 decision(confirmed/rejected/uncertain), event_type(简短英文或null), "
        "event_subtype(简短英文或null), outcome(简短英文或null), contact_time_sec(数字或null), "
        "evidence_frame_indices(从0开始的整数数组), confidence(0到1), uncertainty_reason(中文或null), "
        "short_reason_zh(中文短句), observation_zh(可见动作事实), tactical_analysis_zh(基于上述事实的技战术解读，80-160字，"
        "不确定的动作不得作为确定事实继续推理), limitations_zh(画面缺失和无法判断之处)。confidence 是主观证据充分度，非校准概率。"
        "还须返回 ball_out_of_play:{observed:true/false/null,boundary:touchline/goal_line/unknown,evidence_frame_indices:[]},"
        "restart_observed:true/false/null。门将候选还须返回 goalkeeper:{identity_visible:true/false,ball_contact_visible:true/false,"
        "incoming_shot_visible:true/false,diving_motion_visible:true/false,action_type:save/catch/parry/punch/foot_pass/clearance/rush/unknown}。"
        "event_type 只用 shot、set_piece_delivery、goalkeeper_intervention、defensive_intervention、pass "
        "或ball_out_of_play；角球/任意球应放在 event_subtype，不要放在 event_type。"
        f"\n候选上下文: {json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
    )
    content = [{"type": "text", "text": instructions}]
    for index, frame in enumerate(frames):
        content.append({"type": "text", "text": f"帧 {index}，视频时间 {frame['time_sec']:.3f} 秒"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(frame["jpeg"]).decode("ascii"),
                    "detail": "high",
                },
            }
        )
        if frame.get("focus_jpeg"):
            content.extend([
                {"type": "text", "text": f"帧 {index} 的疑似门将局部图（不增加证据帧编号）"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(frame["focus_jpeg"]).decode("ascii"),
                    "detail": "high",
                }},
            ])
    return {
        "model": "deepseek-flash",
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 1800,
        "messages": [{"role": "user", "content": content}],
    }


def call_vision_api(payload: dict, api_key: str) -> tuple[dict, dict]:
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"DeepSeek API 返回 HTTP {exc.code}；请检查密钥、余额和模型权限") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DeepSeek API 网络连接失败: {exc.reason}") from None
    choice = body["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("视觉模型输出被截断，未保存复核结论")
    content = choice["message"]["content"]
    return json.loads(content), body.get("usage", {})


def validate_review(review: dict, frame_count: int) -> dict:
    if not isinstance(review, dict) or review.get("decision") not in DECISIONS:
        raise ValueError("模型没有返回有效的 decision")
    if isinstance(review.get("confidence"), bool) or not isinstance(review.get("confidence"), (int, float)) or not 0 <= review["confidence"] <= 1:
        raise ValueError("模型没有返回 0-1 范围内的 confidence")
    indices = review.get("evidence_frame_indices")
    if not isinstance(indices, list) or any(
        not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < frame_count for i in indices
    ):
        raise ValueError("模型返回的证据帧索引无效")
    if not isinstance(review.get("short_reason_zh"), str):
        raise ValueError("模型没有返回简短说明")
    if review["decision"] == "confirmed" and not review.get("event_type"):
        raise ValueError("确认事件时必须给出 event_type")
    if review["decision"] == "confirmed" and not indices:
        raise ValueError("确认事件时必须给出证据帧")
    for key in ("observation_zh", "tactical_analysis_zh", "limitations_zh"):
        if key in review and not isinstance(review[key], str):
            raise ValueError(f"{key} 必须为文本")
    out = review.get("ball_out_of_play")
    if out is not None:
        if not isinstance(out, dict) or out.get("observed") not in (True, False, None):
            raise ValueError("出界证据格式无效")
        out_indices = out.get("evidence_frame_indices", [])
        if not isinstance(out_indices, list) or any(type(i) is not int or not 0 <= i < frame_count for i in out_indices):
            raise ValueError("出界证据帧无效")
        if out.get("observed") is True and not out_indices:
            raise ValueError("出界判断缺少证据帧")
    return review


def fusion_status(candidate: dict, review: dict) -> str:
    if is_set_piece(candidate) and (review.get("ball_out_of_play") or {}).get("observed") is True and review.get("restart_observed") is not True:
        return "out_of_play_supported_restart_unconfirmed"
    if review["decision"] == "rejected":
        return "rejected_candidate"
    if review["decision"] == "uncertain":
        return "needs_human_review"
    event_type = str(review.get("event_type", "")).lower()
    event_type = EVENT_ALIASES.get(event_type, event_type)
    expected = EVENT_NAMES[candidate["event_type"]]
    if event_type not in {expected, candidate["event_type"]}:
        return "conflict_needs_human_review"
    if is_set_piece(candidate):
        if review.get("restart_observed") is False:
            return "needs_human_review"
        raw_type = str(review.get("event_type") or "").lower()
        raw_subtype = str(review.get("event_subtype") or (raw_type if raw_type in SET_PIECE_SUBTYPES else "")).lower()
        visual_subtype = SET_PIECE_ALIASES.get(raw_subtype, raw_subtype)
        expected_subtype = str(candidate.get("event_subtype") or candidate.get("metrics", {}).get("set_piece_type") or "unknown").lower()
        if visual_subtype not in SET_PIECE_SUBTYPES:
            return "needs_human_review"
        if expected_subtype in SET_PIECE_SUBTYPES and visual_subtype != expected_subtype:
            return "conflict_needs_human_review"
    if candidate["event_type"] == "goalkeeper_intervention_candidate":
        keeper = review.get("goalkeeper") or {}
        if keeper.get("identity_visible") is not True or keeper.get("ball_contact_visible") is not True:
            return "needs_human_review"
        if keeper.get("action_type") in {"save", "parry"} and keeper.get("incoming_shot_visible") is not True:
            return "needs_human_review"
    return "visual_supported_candidate"


def save_evidence(video: Path, event: dict, frames: list[dict], directory: Path,
                  before_sec: float = 5.0, after_sec: float = 5.0) -> dict:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    # Reruns may change the sampling density; remove only our generated image names.
    for pattern in ("frame_[0-9][0-9][0-9]_*.jpg", "focus_[0-9][0-9][0-9].jpg"):
        for previous in directory.glob(pattern):
            previous.unlink()
    capture = cv2.VideoCapture(str(video))
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / max(capture.get(cv2.CAP_PROP_FPS), 1e-6)
    capture.release()
    start, end = evidence_bounds(event, duration, before_sec, after_sec)
    image_paths = []
    for index, frame in enumerate(frames):
        filename = f"frame_{index:03d}_{frame['time_sec']:.3f}.jpg"
        (directory / filename).write_bytes(frame["jpeg"])
        image_paths.append(str((directory / filename).resolve()))
        if frame.get("focus_jpeg"):
            (directory / f"focus_{index:03d}.jpg").write_bytes(frame["focus_jpeg"])
    clip = directory / "review_clip.mp4"
    ffmpeg = shutil.which("ffmpeg") or str(Path.home() / "anaconda3/envs/sports/bin/ffmpeg")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-ss", str(start), "-i", str(video), "-t", str(end - start),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
                    "-movflags", "+faststart", str(clip)], check=True)
    return {"clip_path": str(clip.resolve()), "image_paths": image_paths,
            "start_sec": start, "end_sec": end}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=41)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--before-sec", type=float, default=5.0)
    parser.add_argument("--after-sec", type=float, default=5.0)
    parser.add_argument("--gsr-json", type=Path, help="Add same-frame keeper crops from these tracks.")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="只检验候选及证据帧，不调用 API")
    args = parser.parse_args()
    if not 4 <= args.frames <= 61 or not 320 <= args.width <= 1280:
        parser.error("--frames 需在 4-61 之间，--width 需在 320-1280 之间")
    if not 0.1 <= args.before_sec <= 8 or not 0.1 <= args.after_sec <= 8:
        parser.error("--before-sec 和 --after-sec 需在 0.1-8 秒之间")
    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    event = find_candidate(report, args.event_id)
    reported_video = report.get("source", {}).get("video")
    if reported_video and Path(reported_video).resolve() != args.video.resolve():
        parser.error("报告关联的视频与 --video 不一致；请使用对应的原始视频")
    focus_boxes = {}
    if args.gsr_json and event.get("event_type") == "goalkeeper_intervention_candidate":
        from tactical_analysis.gsr_io import parse_frame
        predictions = json.loads(args.gsr_json.read_text(encoding="utf-8")).get("predictions", [])
        focus_boxes = {parse_frame(p["image_id"]): p["bbox_image"] for p in predictions
                       if p.get("track_id") == event.get("actor_track_id") and p.get("bbox_image")}
    frames = extract_frames(args.video, event, args.frames, args.width, args.before_sec, args.after_sec, focus_boxes)
    evidence_dir = args.evidence_dir or args.output_json.parent / f"{args.event_id}.evidence"
    evidence = save_evidence(args.video, event, frames, evidence_dir, args.before_sec, args.after_sec)
    result = {
        "schema_version": "2.0",
        "prompt_version": REVIEW_PROMPT_VERSION,
        "source_report": str(args.report_json.resolve()),
        "source_video": str(args.video.resolve()),
        "candidate": event,
        "evidence_times_sec": [frame["time_sec"] for frame in frames],
        "evidence_window_sec": [frames[0]["time_sec"], frames[-1]["time_sec"]],
        "frame_count": len(frames),
        "focus_frame_count": sum(bool(frame.get("focus_jpeg")) for frame in frames),
        "evidence": evidence,
        "model": "deepseek-flash",
    }
    if args.dry_run:
        result["status"] = "evidence_ready_no_api_call"
    else:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            parser.error("未设置 DEEPSEEK_API_KEY 环境变量")
        review, usage = call_vision_api(build_request(event, frames), api_key)
        result["vision_review"] = validate_review(review, len(frames))
        result["fusion_status"] = fusion_status(event, result["vision_review"])
        result["usage"] = usage
        result["status"] = "reviewed"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入复核结果: {args.output_json}")
    if result.get("fusion_status"):
        print(f"融合状态: {result['fusion_status']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)
