"""Render panoramic or camera-focused views with event and spatial evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FONT = ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"
WIDTH, SOURCE_HEIGHT, HEIGHT = 1280, 272, 720
PANEL_HEIGHT = HEIGHT - SOURCE_HEIGHT
FOCUS_TOP = 552
WHITE = (236, 242, 246)
MUTED = (160, 174, 183)
BLUE = (99, 161, 245)
AMBER = (231, 173, 87)
DISPLAY_GROUPS = {
    "L1-01": "射门与结果", "L1-02": "定位球", "L1-03": "接球、持球与传球",
    "L1-04": "球权转换", "L1-05": "防守动作", "L1-06": "门将事件",
    "L1-07": "比赛重启",
}


@lru_cache(maxsize=16)
def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), size)


def clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def wrapped(draw: ImageDraw.ImageDraw, content: str, max_width: int, size: int) -> list[str]:
    lines, line = [], ""
    for char in content:
        next_line = line + char
        if line and draw.textbbox((0, 0), next_line, font=font(size))[2] > max_width:
            lines.append(line)
            line = char
        else:
            line = next_line
    if line:
        lines.append(line)
    return lines


def panel(report: dict, window: dict, event: dict | None, duration: float,
          review_by_id: dict | None = None, timeline_events: list[dict] | None = None) -> np.ndarray:
    image = Image.new("RGB", (WIDTH, PANEL_HEIGHT), (28, 32, 35))
    draw = ImageDraw.Draw(image)
    has_vision = bool(review_by_id)
    draw.text((26, 18), "双机位技战术复核" if has_vision else "关键事件与空间复盘", font=font(27), fill=WHITE)
    draw.text((877 if has_vision else 945, 26),
              "每节点16时刻 × 双机位" if has_vision else "二维轨迹候选 · 待人工复核",
              font=font(16), fill=AMBER)
    draw.line((24, 67, WIDTH - 24, 67), fill=(62, 71, 77), width=1)
    draw.line((780, 84, 780, 338), fill=(62, 71, 77), width=1)

    draw.text((26, 85), "当前关键事件", font=font(18), fill=BLUE)
    if event:
        draw.text((26, 122), event["label_zh"], font=font(29), fill=WHITE)
        review = (review_by_id or {}).get(event["event_id"])
        if review:
            status = {"visible": "画面可见（非规则确认）", "uncertain": "画面无法确认",
                      "not_visible": "审核画面未见"}.get(review.get("decision"), "模型调用失败")
            draw.text((26, 163), f"{DISPLAY_GROUPS[event['l1_group']]}  /  {clock(event['timestamp_sec'])}  /  {status}",
                      font=font(18), fill=AMBER)
            for index, line in enumerate(wrapped(draw, review.get("observation_zh") or event["summary_zh"], 725, 18)[:3]):
                draw.text((26, 204 + index * 27), line, font=font(18), fill=WHITE)
            draw.text((26, 302), f"{review.get('review_view_count', 0)} 个画面复核 · 仍为事件候选",
                      font=font(15), fill=MUTED)
        else:
            draw.text((26, 163), f"{DISPLAY_GROUPS[event['l1_group']]}  /  {clock(event['timestamp_sec'])}",
                      font=font(18), fill=AMBER)
            for index, line in enumerate(wrapped(draw, event["summary_zh"], 720, 19)[:3]):
                draw.text((26, 205 + index * 29), line, font=font(19), fill=WHITE)
            draw.text((26, 301), f"证据窗 {clock(event['evidence_start_sec'])}–{clock(event['evidence_end_sec'])}  ·  未确认比赛事件",
                      font=font(15), fill=MUTED)
    else:
        draw.text((26, 124), "当前无重点复核节点" if has_vision else "当前无事件候选",
                  font=font(28), fill=WHITE)
        draw.text((26, 177), "时间轴上的节点仅为候选，未被确认不等于事件不存在。" if has_vision
                  else "继续查看下方时间轴；无候选不代表比赛中没有事件。", font=font(18), fill=MUTED)

    blue, red = window["teams"]["blue"], window["teams"]["red"]
    ball = window.get("ball") or {}
    draw.text((810, 85), f"{clock(window['start_sec'])}–{clock(window['end_sec'])} 空间窗口", font=font(18), fill=BLUE)
    draw.text((810, 126), f"蓝队  宽 {blue['width_m']:.1f}m  纵深 {blue['depth_m']:.1f}m", font=font(18), fill=WHITE)
    draw.text((810, 161), f"红队  宽 {red['width_m']:.1f}m  纵深 {red['depth_m']:.1f}m", font=font(18), fill=WHITE)
    draw.text((810, 205), f"足球观测 {ball.get('observed_ball_frames', 0)} 帧", font=font(18), fill=WHITE)
    draw.text((810, 241), f"主要区域 {ball.get('top_zone') or '未知'}", font=font(18), fill=MUTED)
    draw.text((810, 299), "站位仅统计可见球员", font=font(15), fill=MUTED)

    left, right, y = 74, WIDTH - 34, 376
    draw.line((left, y, right, y), fill=(78, 87, 94), width=5)
    for item in timeline_events if timeline_events is not None else report["l1"]["events"]:
        x = left + round((right - left) * item["timestamp_sec"] / duration)
        color = AMBER if item["l1_group"] == "L1-07" else (108, 193, 155) if item["event_type"] == "carry" else BLUE
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
    draw.text((25, 399), "时间轴", font=font(15), fill=MUTED)
    draw.text((650, 399), "近球与叠加轨迹不等于真实触球证据。" if has_vision
              else "近球关系不等于触球；中圈开球不计作角球或任意球。", font=font(15), fill=MUTED)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def fit_region(canvas: np.ndarray, region: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    scale = min(width / region.shape[1], height / region.shape[0])
    fitted_width = round(region.shape[1] * scale)
    fitted_height = round(region.shape[0] * scale)
    target_x = x + (width - fitted_width) // 2
    target_y = y + (height - fitted_height) // 2
    canvas[target_y:target_y + fitted_height, target_x:target_x + fitted_width] = cv2.resize(
        region, (fitted_width, fitted_height), interpolation=cv2.INTER_LINEAR)


def focus_header() -> np.ndarray:
    image = Image.new("RGB", (WIDTH, FOCUS_TOP), (17, 22, 24))
    draw = ImageDraw.Draw(image)
    draw.text((971, 190), "机位二 · 小窗", font=font(19), fill=WHITE)
    draw.text((971, 235), "二维球场 · 同步", font=font(19), fill=WHITE)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render(video: Path, report_path: Path, output: Path, layout: str = "panorama") -> None:
    if layout not in {"panorama", "focus"}:
        raise ValueError(f"Unknown layout: {layout}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review_by_id = {item["event_id"]: item for item in report.get("vision_review", {}).get("results", [])}
    events = [item for item in report["l1"]["events"] if item["event_id"] in review_by_id] if review_by_id else report["l1"]["events"]
    windows = report["windows"]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if (source_width, source_height) != (WIDTH, SOURCE_HEIGHT) or fps <= 0 or frame_count <= 0:
        raise ValueError("Expected a 1280x272 two-camera source with valid FPS and frames")
    duration = frame_count / fps
    if abs(duration - float(report["summary"]["duration_sec"])) > 0.2:
        raise ValueError("Video and L1 report duration do not match")

    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required in the active environment")
    panel_top = SOURCE_HEIGHT if layout == "panorama" else FOCUS_TOP
    output_height = panel_top + PANEL_HEIGHT
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{output_height}",
               "-r", str(fps), "-i", "pipe:0", "-i", str(video),
               "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "fast",
               "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
               "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cache: dict[tuple[int, int], np.ndarray] = {}
    event_times = [item["timestamp_sec"] for item in events]
    window_starts = [item["start_sec"] for item in windows]
    left, right, y = 74, WIDTH - 34, panel_top + 376
    focus_background = focus_header() if layout == "focus" else None
    try:
        for frame_index in range(frame_count):
            ok, original = capture.read()
            if not ok:
                raise RuntimeError(f"Video decoding ended at frame {frame_index}")
            time_sec = frame_index / fps
            window_index = max(0, min(len(windows) - 1, bisect_right(window_starts, time_sec) - 1))
            if review_by_id:
                nearby = [index for index, event_time in enumerate(event_times)
                          if event_time - 2.0 <= time_sec <= event_time + 1.6]
                event_index = min(nearby, key=lambda index: abs(event_times[index] - time_sec)) if nearby else -1
            else:
                event_index = bisect_right(event_times, time_sec) - 1
                if event_index >= 0 and time_sec - event_times[event_index] > 3.0:
                    event_index = -1
            key = (window_index, event_index)
            if key not in cache:
                cache[key] = panel(report, windows[window_index], events[event_index] if event_index >= 0 else None,
                                   duration, review_by_id, events)
            canvas = np.empty((output_height, WIDTH, 3), dtype=np.uint8)
            if layout == "focus":
                canvas[:FOCUS_TOP] = focus_background
                fit_region(canvas, original[:, :472], 0, 0, 960, 540)
                fit_region(canvas, original[:, 472:944], 968, 0, 312, 180)
                fit_region(canvas, original[:, 944:1280], 968, 260, 312, 252)
            else:
                canvas[:SOURCE_HEIGHT] = original
            canvas[panel_top:] = cache[key]
            progress_x = left + round((right - left) * min(1.0, time_sec / duration))
            cv2.line(canvas, (left, y), (progress_x, y), (245, 161, 99), 5, cv2.LINE_AA)
            cv2.circle(canvas, (progress_x, y), 10, (245, 161, 99), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"{clock(time_sec)} / {clock(duration)}", (75, panel_top + 417),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (236, 242, 246), 1, cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
    finally:
        capture.release()
        process.stdin.close()
    error = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {error}")
    poster = output.with_name(output.stem + "_封面.jpg")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", "1",
                    "-i", str(output), "-frames:v", "1", str(poster)], check=True)
    print(f"Rendered {output} ({layout}, {frame_count} frames, {duration:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("二维分析/分析视频2.mp4"))
    parser.add_argument("--report", type=Path, default=Path("二维分析/分析视频2_技战术结果/tactical_analysis_report.json"))
    parser.add_argument("--layout", choices=("panorama", "focus"), default="panorama")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("二维分析/分析视频2_技战术结果") / (
        "分析视频2_全景技战术.mp4" if args.layout == "panorama" else "分析视频2_主机位预览.mp4")
    render(args.video, args.report, output, args.layout)


if __name__ == "__main__":
    main()
