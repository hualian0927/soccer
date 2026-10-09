"""Render a match video with a separate, time-aligned vision-review panel."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FONT_PATH = ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"
CANVAS_WIDTH, CANVAS_HEIGHT = 1600, 900
HEADER_HEIGHT = 72
VIDEO_WIDTH, VIDEO_HEIGHT = 1200, 675
PANEL_WIDTH = CANVAS_WIDTH - VIDEO_WIDTH
FOOTER_TOP = HEADER_HEIGHT + VIDEO_HEIGHT
BACKGROUND = (25, 28, 31)
PANEL = (31, 34, 37)
WHITE = (240, 244, 247)
MUTED = (160, 170, 178)
BLUE = (56, 128, 237)
GREEN = (65, 190, 135)
AMBER = (241, 177, 73)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def clock_time(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def wrap(draw: ImageDraw.ImageDraw, value: str, max_width: int, text_font: ImageFont.ImageFont) -> list[str]:
    lines: list[str] = []
    line = ""
    for char in value:
        candidate = line + char
        if line and draw.textbbox((0, 0), candidate, font=text_font)[2] > max_width:
            lines.append(line)
            line = char
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def load_reviews(directories: Path | list[Path], video: Path) -> list[dict]:
    by_event = {}
    paths = [directories] if isinstance(directories, Path) else directories
    for directory in paths:
        for path in sorted(directory.glob("*.deepseek.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if Path(data["source_video"]).resolve() != video.resolve():
                raise ValueError(f"视觉复核结果对应的视频不同: {path}")
            if data.get("status") != "reviewed" or "vision_review" not in data:
                continue
            event_id = data["candidate"]["event_id"]
            previous = by_event.get(event_id)
            if previous is None or int(data.get("frame_count") or 0) >= int(previous.get("frame_count") or 0):
                by_event[event_id] = data
    reviews = list(by_event.values())
    if not reviews:
        raise ValueError(f"目录中没有可用的视觉复核结果: {paths}")
    reviews.sort(key=lambda item: item["candidate"]["time_sec"])
    return reviews


def review_display_time(review: dict) -> float:
    candidate_time = float(review["candidate"]["time_sec"])
    if review["candidate"]["event_type"] in {"corner_candidate", "set_piece_delivery_candidate"}:
        window = review.get("evidence_window_sec")
        if isinstance(window, list) and len(window) == 2:
            return max(candidate_time, float(window[1]))
    return candidate_time


def review_style(review: dict) -> tuple[str, tuple[int, int, int]]:
    if review.get("fusion_status") == "visual_supported_candidate":
        return "视觉支持 · 候选", GREEN
    if review.get("fusion_status") == "rejected_candidate":
        return "视觉否定 · 待复核", (225, 105, 106)
    if review.get("fusion_status") == "conflict_needs_human_review":
        return "结论冲突 · 待复核", (225, 105, 106)
    return "证据不足 · 待复核", AMBER


def draw_card(
    draw: ImageDraw.ImageDraw, review: dict, y: int, height: int, *, prominent: bool,
    upcoming: bool = False,
) -> None:
    x0, x1 = 18, PANEL_WIDTH - 18
    fill = (44, 51, 57) if prominent else (36, 40, 44)
    draw.rounded_rectangle((x0, y, x1, y + height), radius=7, fill=fill, outline=(58, 65, 72))
    candidate = review["candidate"]
    vision = review["vision_review"]
    badge, accent = ("证据待完整", BLUE) if upcoming else review_style(review)
    draw.rounded_rectangle((x0 + 14, y + 13, x0 + 175, y + 43), radius=6, fill=accent)
    draw.text((x0 + 22, y + 16), badge, font=font(15), fill=(19, 24, 27))
    draw.text((x1 - 80, y + 16), clock_time(candidate["time_sec"]), font=font(17), fill=MUTED)
    title = candidate.get("label_zh") or candidate["event_type"]
    if candidate["event_type"] in {"corner_candidate", "set_piece_delivery_candidate"} and review.get("fusion_status") != "visual_supported_candidate":
        title = "定位球类型待复核"
    draw.text((x0 + 14, y + 55), title, font=font(24 if prominent else 19), fill=WHITE)
    if prominent:
        draw.text((x0 + 14, y + 94), "二维候选  /  多帧画面复核", font=font(16), fill=MUTED)
        reason = vision.get("short_reason_zh") or vision.get("uncertainty_reason") or "暂无视觉说明"
        text_font = font(18)
        for index, line in enumerate(wrap(draw, reason, x1 - x0 - 28, text_font)[:3]):
            draw.text((x0 + 14, y + 125 + index * 29), line, font=text_font, fill=WHITE)
        draw.text((x0 + 14, y + height - 32), "模型判断不能替代比赛官方标签", font=font(14), fill=MUTED)
    else:
        reason = "播放完整证据片段后显示视觉结论" if upcoming else (vision.get("short_reason_zh") or "")
        for index, line in enumerate(wrap(draw, reason, x1 - x0 - 28, font(15))[:2]):
            draw.text((x0 + 14, y + 86 + index * 23), line, font=font(15), fill=MUTED)


def render_panel(reviews: list[dict], active_index: int) -> np.ndarray:
    image = Image.new("RGB", (PANEL_WIDTH, VIDEO_HEIGHT), PANEL)
    draw = ImageDraw.Draw(image)
    draw.text((20, 22), "技战术关键节点", font=font(27), fill=WHITE)
    supported = sum(review_style(review)[0].startswith("视觉支持") for review in reviews)
    draw.text((20, 65), f"{len(reviews)} 个候选  ·  {supported} 个视觉支持", font=font(16), fill=MUTED)
    draw.line((0, 97, PANEL_WIDTH, 97), fill=(55, 60, 65), width=1)
    if active_index < 0:
        draw.text((20, 123), "等待第一个关键节点", font=font(22), fill=WHITE)
        draw.text((20, 158), "播放过程中逐条显示视觉复核结论", font=font(16), fill=MUTED)
        upcoming = reviews[:3]
        draw.text((20, 233), "待复核节点", font=font(19), fill=WHITE)
        y = 269
    else:
        draw_card(draw, reviews[active_index], 113, 229, prominent=True)
        upcoming = reviews[active_index + 1 : active_index + 3]
        draw.text((20, 361), "后续节点", font=font(19), fill=WHITE)
        y = 394
    for review in upcoming:
        draw_card(draw, review, y, 127, prominent=False, upcoming=True)
        y += 138
    if not upcoming and active_index >= 0:
        draw.text((20, 407), "已到达本片最后一个复核节点", font=font(17), fill=MUTED)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_static_canvas(reviews: list[dict], start: float, duration: float) -> np.ndarray:
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, CANVAS_WIDTH, HEADER_HEIGHT), fill=(22, 25, 28))
    draw.rectangle((0, FOOTER_TOP, CANVAS_WIDTH, CANVAS_HEIGHT), fill=(26, 29, 33))
    draw.text((25, 16), "视觉模型技战术复核", font=font(30), fill=WHITE)
    draw.text((390, 26), "二维候选 × 原始画面 × 多帧视觉证据", font=font(19), fill=MUTED)
    draw.text((CANVAS_WIDTH - 345, 25), "实验输出  /  非正式比赛统计", font=font(18), fill=AMBER)
    draw.text((28, FOOTER_TOP + 23), "比赛时间轴", font=font(20), fill=WHITE)
    draw.text((28, FOOTER_TOP + 90), "绿色：视觉支持  ·  黄色：证据不足  ·  红色：结论冲突", font=font(16), fill=MUTED)
    draw.text((CANVAS_WIDTH - 445, FOOTER_TOP + 90), "关键事件需回看原片并由人工确认", font=font(16), fill=MUTED)
    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    y = FOOTER_TOP + 61
    cv2.line(canvas, (205, y), (CANVAS_WIDTH - 35, y), (68, 73, 78), 5, cv2.LINE_AA)
    for review in reviews:
        position = (review["candidate"]["time_sec"] - start) / duration
        x = 205 + round((CANVAS_WIDTH - 240) * position)
        _, color = review_style(review)
        cv2.circle(canvas, (x, y), 7, color[::-1], -1, cv2.LINE_AA)
    return canvas


def render_video(
    video: Path, reviews: list[dict], output: Path, output_fps: int, start: float, duration: float
) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"无法读取视频: {video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if source_fps <= 0 or source_frames <= 0:
        raise ValueError("视频缺少有效帧率或长度")
    available = source_frames / source_fps - start
    duration = min(duration, available)
    if duration <= 0:
        raise ValueError("所选时间段不在视频内")
    reviews = [r for r in reviews if start <= r["candidate"]["time_sec"] < start + duration]
    if not reviews:
        raise ValueError("所选时间段内没有视觉复核节点")
    reviews.sort(key=review_display_time)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + ".silent.tmp.mp4")
    if temp.exists():
        raise FileExistsError(f"临时文件已存在，请先检查: {temp}")
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (CANVAS_WIDTH, CANVAS_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("无法创建视频编码器")
    static = render_static_canvas(reviews, start, duration)
    panel_cache = {index: render_panel(reviews, index) for index in range(-1, len(reviews))}
    event_times = [review_display_time(review) for review in reviews]
    source_index = -1
    frame = None
    total_frames = int(duration * output_fps)
    try:
        for output_index in range(total_frames):
            absolute_time = start + output_index / output_fps
            target_index = min(source_frames - 1, round(absolute_time * source_fps))
            while source_index < target_index:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"视频在 {absolute_time:.2f} 秒解码失败")
                source_index += 1
            canvas = static.copy()
            canvas[HEADER_HEIGHT:FOOTER_TOP, :VIDEO_WIDTH] = cv2.resize(
                frame, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_AREA
            )
            active_index = bisect_right(event_times, absolute_time) - 1
            canvas[HEADER_HEIGHT:FOOTER_TOP, VIDEO_WIDTH:] = panel_cache[active_index]
            progress = max(0.0, min(1.0, (absolute_time - start) / duration))
            x = 205 + round((CANVAS_WIDTH - 240) * progress)
            y = FOOTER_TOP + 61
            cv2.line(canvas, (205, y), (x, y), BLUE[::-1], 5, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 11, WHITE[::-1], -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 7, BLUE[::-1], -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{clock_time(absolute_time)} / {clock_time(start + duration)}",
                (28, FOOTER_TOP + 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                WHITE[::-1],
                2,
                cv2.LINE_AA,
            )
            writer.write(canvas)
            if output_index and output_index % (output_fps * 30) == 0:
                print(f"已渲染 {output_index / output_fps:.0f}/{duration:.0f} 秒", flush=True)
    finally:
        writer.release()
        capture.release()
    ffmpeg = shutil.which("ffmpeg") or str(Path.home() / "anaconda3/envs/sports/bin/ffmpeg")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(temp),
        "-ss", str(start), "-t", str(duration), "-i", str(video),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(output),
    ]
    subprocess.run(command, check=True)
    temp.unlink()
    print(f"视频已生成: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reviews-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"输出文件已存在: {args.output}")
    if args.start < 0 or args.duration <= 0 or args.duration > 600 or not 10 <= args.fps <= 30:
        parser.error("参数范围错误：起点需非负，时长在 0-600 秒，帧率在 10-30")
    reviews = load_reviews(args.reviews_dir, args.video)
    render_video(args.video, reviews, args.output, args.fps, args.start, args.duration)


if __name__ == "__main__":
    main()
