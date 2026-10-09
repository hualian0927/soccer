"""Analyze a supplied two-camera projection folder and export a Chinese PDF report."""

from __future__ import annotations

import argparse
import base64
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager

from analyze_full_pitch_projection import aggregate_window, frame_metrics, median
from analyze_projection_with_ball import (
    ball_window_metrics,
    center_restart_candidates,
    outside_candidates,
    player_index,
)
from review_tactical_candidates_with_vision import call_vision_api
from tactical_analysis.projection_l1 import analyze_projection_l1


ROOT = Path(__file__).resolve().parent
FONT_PATH = ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"
PAGE = (11.69, 8.27)
BG = "#f8faf9"
INK = "#172a28"
MUTED = "#536563"
BLUE = "#2d6cb3"
RED = "#c84a4a"
GREEN = "#21926a"


def load_folder(folder: Path) -> tuple[Path, dict, dict, float, int]:
    videos = sorted(folder.glob("*.mp4"))
    if len(videos) != 1:
        raise ValueError(f"Expected one MP4 in {folder}, found {len(videos)}")
    ball = json.loads((folder / "ball_positions.json").read_text(encoding="utf-8"))
    players = json.loads((folder / "player_positions.json").read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(videos[0]))
    if not capture.isOpened():
        raise ValueError(f"Cannot read video: {videos[0]}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if abs(fps - float(ball["meta"]["fps"])) > 0.01 or frames != int(ball["meta"]["frames_range"][-1]):
        raise ValueError("Video and ball coordinate timelines do not match")
    if frames != int(players["meta"]["frames_range"][-1]):
        raise ValueError("Video and player coordinate timelines do not match")
    return videos[0], ball, players, fps, frames


def spatial_rows(people: list[dict]) -> pd.DataFrame:
    rows = (
        (point["frame"], point["time_s"], person["camera"], person["person_id"],
         person["role"], person["color"], point["x_m"], point["y_m"])
        for person in people for point in person["positions"]
    )
    return pd.DataFrame(rows, columns=["frame", "time_s", "camera", "track_id", "role", "color", "x_m", "y_m"])


def build_report(folder: Path, window_sec: int = 10) -> dict:
    video, ball_data, player_data, fps, frames = load_folder(folder)
    pitch = ball_data["meta"]["pitch"]
    width, length = float(pitch["width_m"]), float(pitch["length_m"])
    duration = frames / fps
    people = player_data["people"]
    positions = ball_data["positions"]
    records = frame_metrics(spatial_rows(people), width, length)
    by_frame = player_index(people)
    windows = []
    for index, start in enumerate(np.arange(0, duration, window_sec)):
        end = min(float(start + window_sec), duration)
        window = aggregate_window(records, index, float(start), end)
        window["ball"] = ball_window_metrics(positions, by_frame, float(start), end, width, length)
        windows.append(window)
    outside = outside_candidates(positions, width, length, fps)
    restarts = center_restart_candidates(positions, width, length, duration)
    l1 = analyze_projection_l1(positions, people, restarts, outside, duration, fps)
    observed = [point for point in positions if point.get("observed")]
    near = Counter()
    for window in windows:
        near.update(window["ball"]["nearest_team_within_3m_frames"])
    by_type = Counter(event["event_type"] for event in l1["events"])
    return {
        "schema_version": "projection-folder-report-v1",
        "source": {"folder": str(folder.resolve()), "video": str(video.resolve()),
                   "ball_json": str((folder / "ball_positions.json").resolve()),
                   "player_json": str((folder / "player_positions.json").resolve())},
        "pitch": {"width_m": width, "length_m": length},
        "summary": {
            "duration_sec": duration, "fps": fps, "frames": frames,
            "ball_observed_frames": len(observed),
            "ball_observed_ratio": round(len(observed) / frames, 3),
            "player_track_fragments": len(people),
            "observed_player_positions": sum(len(person["positions"]) for person in people),
            "team_shape_medians": {
                color: {metric: median([window["teams"][color][metric] for window in windows
                                        if window["teams"][color][metric] is not None])
                        for metric in ("visible_players", "width_m", "depth_m")}
                for color in ("blue", "red")
            },
            "near_ball_frames": dict(near),
            "l1_candidate_count": len(l1["events"]),
            "candidate_counts_by_type": dict(by_type),
            "confirmed_offside_count": 0,
            "confirmed_goal_count": 0,
        },
        "windows": windows,
        "outside_evidence": outside,
        "restart_candidates": restarts,
        "l1": l1,
        "vision_review": {"status": "not_requested", "model": None, "results": []},
        "interpretation": [
            "0-7 秒足球在中圈附近静止，约 6.68 秒离开；这是开球候选，不是已核准的触球时间。",
            "约 41-47 秒出现三次蓝队同机位近球者接续，可作为传球链/踢球动作复核片段；未确认三次成功传球。",
            "约 38-40 秒、52-54 秒蓝队出现持续近球移动，可能是持球推进，仍须核查球是否受控。",
            "该片没有足够证据确认越位：必须同时核准传球瞬间、进攻方向、球与倒数第二名防守队员及接球参与。",
            "球员轨迹 ID 仅在各自机位稳定，不能把 51 个轨迹片段当成 51 名独立球员。",
        ],
        "limitations": [
            "球场单应投影约有 1-2 米误差，飞行中的球投影不是地面真实落点。",
            "近球距离不是正式控球率，也不能证明踢球、传球完成或持球突破。",
            "红队有两个门将轨迹片段；角色标签/跨机位身份可能误判，门将数量不作正式统计。",
            "无裁判哨声、完整比赛规则上下文和逐帧人工标注；越位、犯规、进球均不做确认统计。",
            "仅有 60 秒样本，队形宽度和纵深是可见球员的采样量，不能外推整场阵型。",
        ],
    }


def read_frame(capture: cv2.VideoCapture, second: float, fps: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(second * fps))
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Cannot read evidence frame at {second:.2f}s")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sanitize_vision_result(event: dict, result: dict) -> dict:
    """Keep visual overlays from becoming independent proof of a physical action."""
    result = dict(result)
    observation = result.get("observation_zh", "")
    if (event["event_type"] in {"pass", "carry"} and result.get("decision") == "visible"
            and any(word in observation for word in ("黄色轨迹", "轨迹线", "传球轨迹"))
            and not any(word in observation for word in ("脚部触球", "脚触球", "踢球动作"))):
        result["raw_decision"] = result["decision"]
        result["decision"] = "uncertain"
        result["review_note_zh"] = "模型主要引用视频叠加轨迹，未独立看清脚部触球；保留为待复核。"
    return result


def vision_sample_times(event: dict, duration: float, fps: float, count: int = 16) -> list[float]:
    """Span the likely action, not just the candidate's (often delayed) timestamp."""
    center = float(event["timestamp_sec"])
    kind = event["event_type"]
    if kind == "pass":
        release = center - float(event.get("attributes", {}).get("transfer_gap_sec") or 0.8)
        start, end = release - 0.65, center + 0.55
    elif kind == "carry":
        movement = float(event.get("attributes", {}).get("duration_sec") or 1.0)
        start, end = center - 0.45, center + movement + 0.45
    else:
        start, end = center - 2.0, center + 2.0
    first = max(0, round(start * fps))
    last = min(int(duration * fps) - 1, round(end * fps))
    if last - first + 1 < count:
        raise ValueError(f"Candidate {event['event_id']} has fewer than {count} frames")
    frames = np.rint(np.linspace(first, last, count)).astype(int).tolist()
    return [round(frame / fps, 3) for frame in frames]


def camera_frame(image: np.ndarray, camera_name: str) -> np.ndarray:
    return image[:, :472] if camera_name == "1-430" else image[:, 472:944]


def encode_jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode evidence image")
    return encoded.tobytes()


def vision_review(report: dict, evidence_dir: Path) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        report["vision_review"]["status"] = "not_called_no_api_key"
        return
    video = Path(report["source"]["video"])
    fps = report["summary"]["fps"]
    ball_positions = json.loads(Path(report["source"]["ball_json"]).read_text(encoding="utf-8"))["positions"]
    ball_by_frame = {int(item["frame"]): item for item in ball_positions}
    selected = [event for event in report["l1"]["events"]
                if event["event_type"] in {"restart", "carry", "pass"}]
    evidence_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    results = []
    try:
        for event in selected:
            times = vision_sample_times(event, report["summary"]["duration_sec"], fps)
            event_ball = ball_by_frame.get(int(event["frame"]))
            primary_camera = (event_ball or {}).get("source_camera") or "1-430"
            camera_order = [primary_camera, "2-430" if primary_camera == "1-430" else "1-430"]
            captures = {camera_name: [] for camera_name in camera_order}
            for second in times:
                image = read_frame(capture, second, fps)
                for camera_name in camera_order:
                    captures[camera_name].append(camera_frame(image, camera_name))
            content = [{"type": "text", "text": (
                "你是足球视频证据复核员。两组固定机位画面各有四张四格图，"
                "每组按相同时间顺序展示16帧，格内编号1-16；不要把不同机位当作连续镜头。"
                "随后是主机位动作前后两张大图和二维球场参照。请比较动作前、中、后。"
                "视频中已有的黄色轨迹线、检测框、二维轨迹都不是独立证据；"
                "只有能在真实机位画面中看清球员与球的物理关系，才可判visible。"
                "传球要看出球及接球，持球要看连续控制，开球要看重启动作。"
                "若球太小或被遮挡，必须回答uncertain；越位、进球、犯规不可仅凭点位确认。"
                "只返回 JSON: decision ('visible'/'uncertain'/'not_visible'), "
                "observation_zh (不超过90字), limitation_zh (不超过60字), "
                "key_frame_indices (证据格编号1-16的整数数组, 若不清晰则空数组)。"
                f"候选: {event['label_zh']}，标记时间 {event['timestamp_sec']:.2f}秒；"
                f"取样范围 {times[0]:.2f}-{times[-1]:.2f}秒。"
            )}]
            paths = []
            for camera_name in camera_order:
                for sheet_index in range(4):
                    sheet = np.full((584, 944, 3), 16, dtype=np.uint8)
                    for local_index in range(4):
                        index = sheet_index * 4 + local_index
                        image = captures[camera_name][index]
                        x = (local_index % 2) * 472
                        y = (local_index // 2) * 292
                        sheet[y + 20:y + 292, x:x + 472] = image
                        label = f"{index + 1:02d}  {times[index]:05.2f}s  {camera_name}"
                        cv2.putText(sheet, label, (x + 7, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.43, (235, 235, 235), 1, cv2.LINE_AA)
                    path = evidence_dir / f"{event['event_id']}_{camera_name}_sheet_{sheet_index + 1}.jpg"
                    encoded = encode_jpeg(sheet)
                    path.write_bytes(encoded)
                    paths.append(str(path.resolve()))
                    content.extend([{"type": "text", "text":
                                     f"{camera_name}固定机位，序列图 {sheet_index + 1}/4，格 {sheet_index * 4 + 1}-{sheet_index * 4 + 4}"},
                                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                     + base64.b64encode(encoded).decode("ascii"), "detail": "high"}}])
            for label, index in (("动作前", 5), ("动作后", 10)):
                zoom = cv2.resize(captures[primary_camera][index], (944, 544), interpolation=cv2.INTER_LINEAR)
                encoded = encode_jpeg(zoom)
                content.extend([{"type": "text", "text": f"{label}放大帧，格 {index + 1}，{times[index]:.2f}秒"},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                 + base64.b64encode(encoded).decode("ascii"), "detail": "high"}}])
            center_image = read_frame(capture, event["timestamp_sec"], fps)[:, 944:]
            pitch_encoded = encode_jpeg(center_image)
            content.extend([{"type": "text", "text": "候选时刻的二维球场图，仅供空间参照"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                             + base64.b64encode(pitch_encoded).decode("ascii"), "detail": "high"}}])
            payload = {"model": "deepseek-flash", "thinking": {"type": "disabled"},
                       "response_format": {"type": "json_object"}, "temperature": 0.1,
                       "max_tokens": 500, "messages": [{"role": "user", "content": content}]}
            try:
                answer, usage = call_vision_api(payload, api_key)
                result = {"event_id": event["event_id"], "timestamp_sec": event["timestamp_sec"],
                          "decision": answer.get("decision") if answer.get("decision") in
                          {"visible", "uncertain", "not_visible"} else "uncertain",
                          "observation_zh": str(answer.get("observation_zh") or "")[:100],
                          "limitation_zh": str(answer.get("limitation_zh") or "")[:80],
                          "key_frame_indices": [item for item in answer.get("key_frame_indices", [])
                                                if isinstance(item, int) and 1 <= item <= len(times)],
                          "usage": usage}
            except (RuntimeError, ValueError, KeyError) as exc:
                result = {"event_id": event["event_id"], "status": "api_error", "error": str(exc)}
            result.update({"review_frame_count": len(times), "review_view_count": len(times) * 2,
                           "sampled_times_sec": times, "camera_order": camera_order,
                           "evidence_sheet_paths": paths})
            results.append(sanitize_vision_result(event, result))
    finally:
        capture.release()
    succeeded = sum("decision" in item for item in results)
    status = "called" if succeeded == len(results) else "called_partial" if succeeded else "api_failed"
    report["vision_review"] = {"status": status, "model": "deepseek-flash",
                               "sampling": "16 synchronized timestamps from each of two fixed cameras, eight 2x2 sheets, two enlarged frames, one pitch reference",
                               "results": results}


def markdown_report(report: dict) -> str:
    s = report["summary"]
    rows = ["# 双机位足球技战术分析报告", "",
            f"素材：`{report['source']['video']}`；时长 {s['duration_sec']:.0f} 秒。", "",
            "## 总体观察", "",
            f"- 足球真实检测 {s['ball_observed_frames']}/{s['frames']} 帧（{s['ball_observed_ratio']:.1%}）。",
            f"- 蓝队近球 {s['near_ball_frames'].get('blue', 0)} 帧，红队近球 {s['near_ball_frames'].get('red', 0)} 帧；这不是控球率。",
            f"- L1 事件候选 {s['l1_candidate_count']} 个，包含近球、持球移动、传球与开球等；确认数为 0。",
            "- 越位：无可确认事件；不能把未检出解释为比赛中不存在越位。", "",
            "## 整体空间结构", "",
            "| 时间 | 蓝队可见人数 | 蓝队宽/纵深(m) | 红队可见人数 | 红队宽/纵深(m) | 球观测帧 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for window in report["windows"]:
        blue, red = window["teams"]["blue"], window["teams"]["red"]
        rows.append(f"| {window['start_sec']:.0f}-{window['end_sec']:.0f}s | "
                    f"{blue['visible_players']:.0f} | {blue['width_m']:.1f}/{blue['depth_m']:.1f} | "
                    f"{red['visible_players']:.0f} | {red['width_m']:.1f}/{red['depth_m']:.1f} | "
                    f"{window['ball']['observed_ball_frames']} |")
    rows.extend(["", "## 战术解读", ""] + [f"- {item}" for item in report["interpretation"]])
    rows.extend(["", "## 局部动作与事件候选", "",
                 "| 时间 | 动作 | 球队 | 证据和限制 |", "| --- | --- | --- | --- |"])
    for event in report["l1"]["events"]:
        if event["event_type"] in {"touch", "pass", "carry", "restart", "receive"}:
            rows.append(f"| {event['timestamp_sec']:.2f}s | {event['label_zh']} | "
                        f"{event['team_id'] or '待定'} | {event['summary_zh']} |")
    rows.extend(["", "## 越位判读", "", report["interpretation"][3], "",
                 "## 视觉模型复核", "",
                 f"状态：`{report['vision_review']['status']}`；模型："
                 f"`{report['vision_review']['model'] or '未调用'}`。", ""])
    if report["vision_review"].get("sampling"):
        rows.append("取帧方式：每节点 16 个同步时刻 × 双固定机位，共 32 个画面，八张四格序列图，另附两张局部大图与二维球场参照。")
        rows.append("")
    for item in report["vision_review"]["results"]:
        rows.append(f"- {item.get('timestamp_sec', '')}s: {item.get('decision', item.get('status'))}；"
                    f"{item.get('observation_zh', item.get('error', ''))}"
                    f"{('；' + item['review_note_zh']) if item.get('review_note_zh') else ''}")
        if item.get("sampled_times_sec"):
            rows.append(f"  - 帧位：{item['sampled_times_sec'][0]:.2f}-{item['sampled_times_sec'][-1]:.2f}s，"
                        f"共 {item['review_view_count']} 个双机位画面；模型指定证据格：{item.get('key_frame_indices') or '无'}。")
            rows.append(f"  - 序列图：`{Path(item['evidence_sheet_paths'][0]).parent}`。")
    rows.extend(["", "## 证据限制", ""] + [f"- {item}" for item in report["limitations"]])
    return "\n".join(rows) + "\n"


def page(title: str, kicker: str, number: int):
    fig = plt.figure(figsize=PAGE, facecolor=BG)
    fig.text(0.055, 0.93, kicker, color=GREEN, fontsize=10, weight="bold")
    fig.text(0.055, 0.865, title, color=INK, fontsize=24, weight="bold")
    fig.text(0.055, 0.045, "双机位  /  视觉与二维坐标联合证据  /  候选结论需复核", color=MUTED, fontsize=8)
    fig.text(0.94, 0.045, f"{number:02d}", color=MUTED, fontsize=9, ha="right")
    return fig


def text(fig, x, y, value, size=11, color=INK, **kwargs):
    fig.text(x, y, value, fontsize=size, color=color, va="top", **kwargs)


def pdf_report(report: dict, output: Path) -> None:
    fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    s = report["summary"]
    windows = report["windows"]
    ball = json.loads(Path(report["source"]["ball_json"]).read_text(encoding="utf-8"))["positions"]
    with PdfPages(output) as pdf:
        fig = page("一分钟比赛片段：整体与局部", "分析报告  /  01 概览", 1)
        text(fig, 0.055, 0.79, f"素材时长 {s['duration_sec']:.0f}s     足球检测覆盖 {s['ball_observed_ratio']:.1%}     事件候选 {s['l1_candidate_count']} 条", 13)
        text(fig, 0.055, 0.73, "红蓝两队的空间站位可量化；踢球、传球、持球与开球仅列为候选。", 12)
        ax = fig.add_axes([0.08, 0.30, 0.58, 0.32], facecolor="white")
        obs = [row for row in ball if row.get("observed")]
        ax.scatter([row["time_s"] for row in obs], [row["y_m"] for row in obs],
                   s=3, alpha=0.55, color=BLUE, rasterized=True)
        ax.set_xlim(0, s["duration_sec"])
        ax.set_ylim(-5, report["pitch"]["length_m"] + 5)
        ax.set_xlabel("视频时间（秒）")
        ax.set_ylabel("足球纵向位置（米）")
        ax.grid(alpha=0.15)
        text(fig, 0.70, 0.62, "关键结论", 13, GREEN, weight="bold")
        text(fig, 0.70, 0.56, "约 6.68s  中圈开球候选", 11)
        text(fig, 0.70, 0.50, "41-47s   蓝队三次接续候选", 11)
        text(fig, 0.70, 0.44, "51-54s   持球移动候选", 11)
        text(fig, 0.70, 0.36, "越位 / 进球：未获确认", 11, RED)
        text(fig, 0.08, 0.21, "图中为逐帧检测到的球场投影点；点位变化不等同于真实触球或射门。", 10, MUTED)
        pdf.savefig(fig); plt.close(fig)

        fig = page("队形与球场区域", "分析报告  /  02 整体结构", 2)
        times = [f"{int(row['start_sec'])}-{int(row['end_sec'])}" for row in windows]
        x = np.arange(len(times))
        ax = fig.add_axes([0.08, 0.46, 0.84, 0.26], facecolor="white")
        ax.bar(x - 0.19, [row["teams"]["blue"]["width_m"] for row in windows], 0.35, label="蓝队宽度", color=BLUE)
        ax.bar(x + 0.19, [row["teams"]["red"]["width_m"] for row in windows], 0.35, label="红队宽度", color=RED)
        ax.set_xticks(x, times)
        ax.set_xlabel("时间窗（秒）")
        ax.set_ylabel("横向宽度（米）")
        ax.legend(frameon=False, ncol=2, loc="upper right")
        text(fig, 0.08, 0.38, "十秒窗口                 蓝队可见/纵深         红队可见/纵深         球观测帧", 10, MUTED)
        for i, row in enumerate(windows):
            blue, red = row["teams"]["blue"], row["teams"]["red"]
            text(fig, 0.08, 0.335 - i * 0.038,
                 f"{times[i]:<8}                         {blue['visible_players']:.0f} 人 / {blue['depth_m']:.1f} 米"
                 f"                   {red['visible_players']:.0f} 人 / {red['depth_m']:.1f} 米"
                 f"                   {row['ball']['observed_ball_frames']} 帧", 9)
        text(fig, 0.08, 0.075, "宽度、纵深是可见球员的投影中位数；不等同整场阵型或完整 11 人站位。", 9, MUTED)
        pdf.savefig(fig); plt.close(fig)

        fig = page("关键时刻：原画面与二维球场", "分析报告  /  03 局部证据", 3)
        events = report["l1"]["events"]
        selected = []
        for kind in ("restart", "carry", "pass"):
            selected.extend([item for item in events if item["event_type"] == kind][:1])
        capture = cv2.VideoCapture(report["source"]["video"])
        ball_by_frame = {int(item["frame"]): item for item in ball}
        try:
            for i, event in enumerate(selected):
                y = 0.68 - i * 0.205
                frame = read_frame(capture, event["timestamp_sec"], s["fps"])
                ball_point = ball_by_frame.get(int(event["frame"]))
                camera_name = ball_point.get("source_camera") if ball_point else None
                if camera_name is None:
                    nearby = min(ball, key=lambda item: abs(item["time_s"] - event["timestamp_sec"]))
                    camera_name = nearby.get("source_camera")
                camera = frame[:, :472] if camera_name == "1-430" else frame[:, 472:944]
                pitch_frame = frame[:, 944:]
                ax1 = fig.add_axes([0.06, y - 0.055, 0.30, 0.17]); ax1.imshow(camera); ax1.axis("off")
                ax2 = fig.add_axes([0.37, y - 0.055, 0.17, 0.17]); ax2.imshow(pitch_frame); ax2.axis("off")
                text(fig, 0.56, y + 0.10, f"{event['timestamp_sec']:.2f}s  {event['label_zh']}", 12, GREEN, weight="bold")
                text(fig, 0.56, y + 0.055, ("红队" if event["team_id"] == "red" else
                      "蓝队" if event["team_id"] == "blue" else "双方") + " · 视频与轨迹候选", 10)
                description = {
                    "restart": "中圈静止后球离开；需回看实际开球人。",
                    "carry": "球员与球持续接近且共同位移；控球待复核。",
                    "pass": "踢球/传球候选；脚部触球与接球待复核。",
                }[event["event_type"]]
                text(fig, 0.56, y + 0.012, description, 9, MUTED)
        finally:
            capture.release()
        text(fig, 0.06, 0.105, "证据帧只提供定位；动作确认应查看关键时刻前后的连续视频。", 10, MUTED)
        pdf.savefig(fig); plt.close(fig)

        fig = page("动作时间轴与判读边界", "分析报告  /  04 复核清单", 4)
        interesting = [item for item in events if item["event_type"] in {"restart", "carry", "pass"}]
        text(fig, 0.06, 0.77, "时间        候选动作              球队       证据窗口", 11, GREEN, weight="bold")
        for i, event in enumerate(interesting[:9]):
            team_zh = {"blue": "蓝队", "red": "红队"}.get(event["team_id"], "待定")
            text(fig, 0.06, 0.72 - 0.046 * i,
                 f"{event['timestamp_sec']:>5.2f}s      {event['label_zh']:<14}     "
                 f"{team_zh:<8}     {event['evidence_start_sec']:.1f}-{event['evidence_end_sec']:.1f}s", 10)
        text(fig, 0.06, 0.275, "战术链：41-47s 三次同队近球者接续；可复核踢球、接球及连续配合，不能视为已确认传球。", 10)
        text(fig, 0.06, 0.237, "越位：无确认结果。需核准传球瞬间、进攻方向、球和倒数第二名防守队员位置，", 10, RED)
        text(fig, 0.06, 0.208, "并证明越位位置球员参与进攻；单帧投影不满足这些条件。", 10, RED)
        vision = report["vision_review"]
        status = {"called": "已调用", "called_partial": "部分调用成功", "api_failed": "调用失败",
                  "not_called_no_api_key": "未调用（未配置 DEEPSEEK_API_KEY）"}.get(vision["status"], "未请求")
        text(fig, 0.06, 0.170, f"视觉模型：{status}。模型复核仍不是裁判或人工真值。", 10)
        text(fig, 0.06, 0.132, "方法：双机位球员地面坐标 → 跨镜头空间聚合 → 球轨迹/近球片段 → 候选事件 → 画面核查。", 9, MUTED)
        text(fig, 0.06, 0.101, "限制：投影误差约 1-2 米；跨镜头 ID 不连通；飞行球落点不等于地面落点。", 9, MUTED)
        pdf.savefig(fig); plt.close(fig)

        if vision["results"]:
            fig = page("视觉模型复核记录", "分析报告  /  05 视觉证据", 5)
            if vision.get("sampling"):
                text(fig, 0.06, 0.825, "每节点：16 个时刻 × 双机位 = 32 个画面；另附局部大图与二维球场参照", 10, MUTED)
            text(fig, 0.06, 0.78, "时间与候选                 复核意见                  画面观察", 11, GREEN, weight="bold")
            by_id = {item["event_id"]: item for item in report["l1"]["events"]}
            for index, item in enumerate(vision["results"][:6]):
                event = by_id[item["event_id"]]
                y = 0.73 - index * 0.095
                decision = {"visible": "画面可见", "uncertain": "无法确认", "not_visible": "画面未见"}.get(
                    item.get("decision"), "调用失败")
                text(fig, 0.06, y, f"{event['timestamp_sec']:05.2f}s  {event['label_zh']}", 10)
                text(fig, 0.31, y, decision, 10, GREEN if decision == "画面可见" else MUTED)
                observation = item.get("observation_zh") or item.get("error") or "无有效模型结果"
                text(fig, 0.49, y, observation[:42], 9)
                if len(observation) > 42:
                    text(fig, 0.49, y - 0.03, observation[42:82], 9)
                if item.get("review_note_zh"):
                    text(fig, 0.49, y - 0.058, item["review_note_zh"][:45], 8, RED)
            text(fig, 0.06, 0.12, "模型只做片段复核；未见或不确定不等于比赛事件不存在，画面可见也不等于规则判罚成立。", 9, MUTED)
            pdf.savefig(fig); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("二维分析/分析视频3"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--vision", action="store_true", help="Review selected candidates with DeepSeek when DEEPSEEK_API_KEY is set")
    parser.add_argument("--reuse-vision-json", type=Path,
                        help="Reuse earlier API results when revising the report without making new calls")
    args = parser.parse_args()
    if args.vision and args.reuse_vision_json:
        parser.error("--vision and --reuse-vision-json cannot be used together")
    output_dir = args.output_dir or args.folder / "技战术分析结果"
    report = build_report(args.folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.vision:
        vision_review(report, output_dir / "vision_evidence")
    elif args.reuse_vision_json:
        previous = json.loads(args.reuse_vision_json.read_text(encoding="utf-8"))
        if Path(previous["source"]["video"]).resolve() != Path(report["source"]["video"]).resolve():
            parser.error("Reused vision results must belong to the same source video")
        events = {item["event_id"]: item for item in report["l1"]["events"]}
        review = previous["vision_review"]
        if any(item["event_id"] not in events for item in review["results"]):
            parser.error("Event IDs in reused vision report do not match current analysis")
        review["results"] = [sanitize_vision_result(events[item["event_id"]], item)
                             for item in review["results"]]
        report["vision_review"] = review
    (output_dir / "技战术分析报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "技战术分析报告.md").write_text(markdown_report(report), encoding="utf-8")
    pdf_report(report, output_dir / "技战术分析报告.pdf")
    print(json.dumps({"output_dir": str(output_dir), "summary": report["summary"],
                      "vision_status": report["vision_review"]["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
