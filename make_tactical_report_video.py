#!/usr/bin/env python3
"""Turn a tactical-analysis report and its base visualization into a Chinese review video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
WHITE = (246, 248, 250)
MUTED = (188, 198, 207)
CYAN = (64, 205, 228)
YELLOW = (250, 194, 61)
RED = (245, 92, 87)
GREEN = (70, 203, 132)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True, help="Tactical base visualization video")
    parser.add_argument("--report-json", type=Path, required=True, help="Structured tactical report")
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--title", default="0827-测试一")
    parser.add_argument("--chapter-seconds", type=float, default=5.0)
    parser.add_argument("--title-seconds", type=float, default=2.5)
    parser.add_argument("--summary-seconds", type=float, default=5.0)
    parser.add_argument(
        "--interleave-chapters",
        action="store_true",
        help="Insert each analysis chapter immediately after its source timestamp.",
    )
    return parser.parse_args()


_FONTS: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size not in _FONTS:
        if FONT_PATH.exists():
            _FONTS[size] = ImageFont.truetype(str(FONT_PATH), size=size)
        else:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def text_width(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.ImageFont) -> float:
    return float(draw.textbbox((0, 0), text, font=font_obj)[2])


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and text_width(draw, candidate, font_obj) > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        lines.append(current)
    return lines


def draw_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
    max_width: int,
    line_gap: int = 8,
) -> int:
    font_obj = font(size)
    x, y = xy
    line_height = size + line_gap
    for line in wrap_text(draw, text, font_obj, max_width):
        draw.text((x, y), line, font=font_obj, fill=color)
        y += line_height
    return y


def rgb_frame(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def bgr_frame(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def add_full_overlay(image: Image.Image, color: tuple[int, int, int, int]) -> None:
    image.alpha_composite(Image.new("RGBA", image.size, color))


def add_timeline(draw: ImageDraw.ImageDraw, width: int, height: int, progress: float) -> None:
    y = height - 8
    draw.rectangle((0, y, width, height), fill=(30, 35, 41, 255))
    draw.rectangle((0, y, int(width * max(0.0, min(1.0, progress))), height), fill=CYAN + (255,))


def render_title(frame: np.ndarray, title: str, source_name: str) -> np.ndarray:
    background = cv2.GaussianBlur(frame, (0, 0), 12)
    image = rgb_frame(background).convert("RGBA")
    add_full_overlay(image, (10, 14, 18, 172))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle((72, 125, 82, 455), fill=CYAN + (255,))
    draw.text((112, 128), title, font=font(54), fill=WHITE + (255,))
    draw.text((112, 207), "足球比赛基础技战术分析测试", font=font(34), fill=CYAN + (255,))
    draw.text((112, 284), "球员与足球追踪  ·  战术板映射  ·  候选事件复核", font=font(24), fill=WHITE + (255,))
    draw.text((112, 349), f"测试素材：{source_name}", font=font(21), fill=MUTED + (255,))
    draw.text((112, 391), "输出原则：候选事件与趋势分析，不等同于比赛官方统计", font=font(21), fill=MUTED + (255,))
    draw.text((112, height - 82), "阶段 1 / 3  数据识别与全景回放", font=font(20), fill=WHITE + (255,))
    return bgr_frame(image.convert("RGB"))


def render_playback(frame: np.ndarray, title: str, elapsed: float, duration: float) -> np.ndarray:
    image = rgb_frame(frame).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle((0, 0, width, 47), fill=(12, 17, 22, 205))
    draw.rectangle((0, 0, 8, 47), fill=CYAN + (255,))
    draw.text((24, 10), f"{title}  ·  全景检测与战术板回放", font=font(21), fill=WHITE + (255,))
    draw.text((width - 190, 11), f"{elapsed:05.1f}s / {duration:05.1f}s", font=font(18), fill=MUTED + (255,))
    add_timeline(draw, width, height, elapsed / max(duration, 0.001))
    return bgr_frame(image.convert("RGB"))


def render_chapter(
    frame: np.ndarray,
    chapter: int,
    title: str,
    summary: str,
    points: list[str],
    confidence: float | None,
    source_time: float | None,
    accent: tuple[int, int, int] = CYAN,
) -> np.ndarray:
    image = rgb_frame(frame).convert("RGBA")
    width, height = image.size
    image.alpha_composite(Image.new("RGBA", image.size, (7, 10, 14, 62)))
    panel_width = int(width * 0.59)
    panel = Image.new("RGBA", (panel_width, height), (9, 14, 19, 225))
    image.alpha_composite(panel, (0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 10, height), fill=accent + (255,))
    draw.text((38, 30), f"分析 {chapter:02d}", font=font(19), fill=accent + (255,))
    if source_time is not None:
        draw.text((panel_width - 150, 31), f"画面 {source_time:.2f}s", font=font(17), fill=MUTED + (255,))
    y = draw_lines(draw, title, (38, 72), 35, WHITE, panel_width - 76, 10)
    y += 22
    y = draw_lines(draw, summary, (38, y), 23, (223, 229, 234), panel_width - 76, 9)
    y += 25
    for point in points:
        draw.ellipse((39, y + 8, 49, y + 18), fill=accent + (255,))
        y = draw_lines(draw, point, (65, y), 20, WHITE, panel_width - 104, 8) + 9
    if confidence is not None:
        conf = max(0.0, min(1.0, confidence))
        bar_x, bar_y, bar_w = 39, height - 83, panel_width - 78
        draw.text((bar_x, bar_y - 30), f"候选置信度  {conf * 100:.0f}%", font=font(17), fill=MUTED + (255,))
        draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + 7), fill=(58, 67, 75, 255))
        draw.rectangle((bar_x, bar_y, bar_x + int(bar_w * conf), bar_y + 7), fill=accent + (255,))
    draw.text((width - 276, height - 43), "阶段 2 / 3  关键片段复核", font=font(17), fill=WHITE + (230,))
    return bgr_frame(image.convert("RGB"))


def render_slow_motion(frame: np.ndarray, elapsed: float, duration: float) -> np.ndarray:
    image = rgb_frame(frame).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle((0, 0, width, 47), fill=(12, 17, 22, 205))
    draw.rectangle((0, 0, 8, 47), fill=YELLOW + (255,))
    draw.text((24, 10), "射门候选动作窗口回放  ·  0.5x", font=font(21), fill=WHITE + (255,))
    draw.text((width - 185, 11), f"窗口 {elapsed:04.1f}s", font=font(18), fill=MUTED + (255,))
    add_timeline(draw, width, height, elapsed / max(duration, 0.001))
    return bgr_frame(image.convert("RGB"))


def render_summary(frame: np.ndarray, title: str, points: list[str]) -> np.ndarray:
    background = cv2.GaussianBlur(frame, (0, 0), 10)
    image = rgb_frame(background).convert("RGBA")
    add_full_overlay(image, (8, 12, 17, 207))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.text((70, 55), f"{title}  ·  测试结论", font=font(40), fill=WHITE + (255,))
    draw.rectangle((70, 116, width - 70, 121), fill=GREEN + (255,))
    y = 162
    for index, point in enumerate(points, start=1):
        draw.text((75, y), f"{index:02d}", font=font(21), fill=GREEN + (255,))
        y = draw_lines(draw, point, (130, y - 2), 23, WHITE, width - 205, 9) + 22
    draw.text((70, height - 68), "阶段 3 / 3  汇总结论", font=font(19), fill=MUTED + (255,))
    return bgr_frame(image.convert("RGB"))


def _formation_team(finding: dict[str, Any] | None) -> str | None:
    title = str((finding or {}).get("title_zh", ""))
    if "左队" in title:
        return "left"
    if "右队" in title:
        return "right"
    return None


def _formation_players(report: dict[str, Any], finding: dict[str, Any] | None) -> list[dict[str, Any]]:
    representative = (finding or {}).get("metrics", {}).get("representative_frame", {})
    frame_number = representative.get("frame")
    team = _formation_team(finding)
    source_json = report.get("source", {}).get("gsr_json")
    if frame_number is None or team is None or not source_json or not Path(source_json).exists():
        return []
    target_suffix = f"{int(frame_number):06d}"
    data = json.loads(Path(source_json).read_text(encoding="utf-8"))
    players = []
    for prediction in data.get("predictions", []):
        if not str(prediction.get("image_id", "")).endswith(target_suffix):
            continue
        attributes = prediction.get("attributes") or {}
        if attributes.get("team") != team or attributes.get("role") != "player":
            continue
        bbox = prediction.get("bbox_image") or {}
        pitch = prediction.get("bbox_pitch") or {}
        if not bbox or pitch.get("x_bottom_middle") is None:
            continue
        players.append(
            {
                "track_id": int(prediction.get("track_id", -1)),
                "x": float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2.0,
                "y": float(bbox.get("y", 0)) + float(bbox.get("h", 0)),
                "pitch_x": float(pitch["x_bottom_middle"]),
            }
        )
    return players


def render_formation_chapter(
    frame: np.ndarray,
    report: dict[str, Any],
    finding: dict[str, Any] | None,
    chapter: int,
) -> np.ndarray:
    output = frame.copy()
    metrics = (finding or {}).get("metrics", {})
    representative = metrics.get("representative_frame", {})
    players = _formation_players(report, finding)
    line_counts = [int(value) for value in representative.get("line_counts", [])]
    direction = 1 if _formation_team(finding) == "left" else -1
    players.sort(key=lambda item: item["pitch_x"] * direction)
    if not line_counts or sum(line_counts) != len(players):
        line_counts = [len(group) for group in np.array_split(np.arange(len(players)), 3) if len(group)]
    accent_bgr = (70, 220, 255)
    offset = 0
    for line_number, count in enumerate(line_counts, 1):
        group = sorted(players[offset: offset + count], key=lambda item: item["x"])
        offset += count
        points = [(int(item["x"]), int(item["y"])) for item in group]
        if len(points) >= 2:
            cv2.polylines(output, [np.asarray(points, dtype=np.int32)], False, accent_bgr, 4, cv2.LINE_AA)
        for point in points:
            cv2.circle(output, point, 7, (12, 18, 24), -1, cv2.LINE_AA)
            cv2.circle(output, point, 5, accent_bgr, -1, cv2.LINE_AA)
        if points:
            cv2.putText(
                output,
                f"L{line_number}",
                (points[0][0], max(24, points[0][1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                accent_bgr,
                2,
                cv2.LINE_AA,
            )

    image = rgb_frame(output).convert("RGBA")
    width, height = image.size
    image.alpha_composite(Image.new("RGBA", (width, 105), (8, 12, 17, 220)), (0, 0))
    image.alpha_composite(Image.new("RGBA", (width, 145), (8, 12, 17, 225)), (0, height - 145))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 10, height), fill=YELLOW + (255,))
    draw.text((34, 22), f"分析 {chapter:02d}  ·  阵型代表帧", font=font(19), fill=YELLOW + (255,))
    draw.text((34, 54), (finding or {}).get("title_zh", "阵型证据不足"), font=font(34), fill=WHITE + (255,))
    summary = (finding or {}).get("summary_zh", "当前画面没有形成稳定阵型证据。")
    draw_lines(draw, summary, (34, height - 118), 22, WHITE, width - 68, 8)
    draw.text(
        (34, height - 40),
        "连线表示该代表帧中的可见后卫、中场与前锋层次，不代表整场固定站位。",
        font=font(18),
        fill=MUTED + (255,),
    )
    return bgr_frame(image.convert("RGB"))


def flatten_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for output in report.get("analyzer_outputs", []):
        findings.extend(output.get("findings", []))
    return findings


def best_finding(findings: list[dict[str, Any]], category: str) -> dict[str, Any] | None:
    matches = [item for item in findings if item.get("category") == category]
    return max(matches, key=lambda item: float(item.get("confidence", 0.0)), default=None)


def evidence_time(finding: dict[str, Any] | None, fallback: float) -> float:
    if finding:
        for evidence in finding.get("evidence", []):
            if evidence.get("time_sec") is not None:
                return float(evidence["time_sec"])
        if finding.get("start_sec") is not None:
            return float(finding["start_sec"])
    return fallback


def video_frame(cap: cv2.VideoCapture, time_sec: float, fallback: np.ndarray) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else fallback.copy()


def metric(finding: dict[str, Any] | None, name: str, default: Any = None) -> Any:
    return (finding or {}).get("metrics", {}).get(name, default)


def write_repeat(writer: cv2.VideoWriter, frame: np.ndarray, count: int) -> None:
    for _ in range(max(1, count)):
        writer.write(frame)


def main() -> None:
    args = parse_args()
    if not args.input_video.exists():
        raise FileNotFoundError(args.input_video)
    if not args.report_json.exists():
        raise FileNotFoundError(args.report_json)

    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    findings = flatten_findings(report)
    quality = best_finding(findings, "data_quality")
    numerical_superiority = best_finding(findings, "numerical_superiority_candidate")
    overload = numerical_superiority or best_finding(findings, "local_overload_candidate")
    pass_event = best_finding(findings, "pass_candidate")
    transition = best_finding(findings, "attacking_transition_candidate") or best_finding(findings, "possession_change")
    high_press_summary = best_finding(findings, "high_press_ppda_summary")
    high_press_episodes = [
        item for item in findings
        if item.get("category") == "pressing_episode_candidate" and metric(item, "zone") == "high_press"
    ]
    pressing = high_press_summary or max(high_press_episodes, key=lambda item: float(item.get("confidence", 0.0)), default=None) or best_finding(findings, "pressing_episode_candidate")
    shot = best_finding(findings, "shot_candidate")
    phase = best_finding(findings, "phase_of_play")
    progression_candidates = [
        item
        for item in findings
        if item.get("category") in {"progressive_pass_candidate", "line_breaking_pass_candidate"}
    ]
    progression = max(progression_candidates, key=lambda item: float(item.get("confidence", 0.0)), default=None)
    shooting_structure = best_finding(findings, "shooting_structure")
    set_piece = best_finding(findings, "set_piece_landing_candidate")
    goalkeeper = best_finding(findings, "goalkeeper_save_candidate")
    attack_chain = best_finding(findings, "attacking_chain_candidate")
    defensive_risk = best_finding(findings, "defensive_third_risk")
    defensive_gap = best_finding(findings, "defensive_gap_candidate")
    pass_network = best_finding(findings, "pass_network_structure")
    formation = best_finding(findings, "formation") or best_finding(findings, "formation_candidate")
    state_shapes = [item for item in findings if item.get("category") == "team_shape_by_possession"]
    shapes = state_shapes or [item for item in findings if item.get("category") == "team_shape"]

    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {args.input_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / max(fps, 1.0)
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Input video contains no readable frames")

    still_cap = cap
    still_first = first_frame
    source_video = Path(report.get("source", {}).get("video", ""))
    if source_video.exists() and source_video.resolve() != args.input_video.resolve():
        clean_cap = cv2.VideoCapture(str(source_video))
        clean_ok, clean_first = clean_cap.read()
        if (
            clean_ok
            and int(clean_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == width
            and int(clean_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == height
        ):
            still_cap = clean_cap
            still_first = clean_first
        else:
            clean_cap.release()

    def still_at(time_sec: float, fallback: np.ndarray | None = None) -> np.ndarray:
        return video_frame(still_cap, time_sec, fallback if fallback is not None else still_first)

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create {args.output_video}")

    source_name = Path(report.get("source", {}).get("video", "720p.mp4")).name
    title_frame = still_at(min(8.0, duration / 2))
    write_repeat(writer, render_title(title_frame, args.title, source_name), round(args.title_seconds * fps))

    if not args.interleave_chapters:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for index in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(render_playback(frame, args.title, index / fps, duration))

    chapter_frames = max(1, round(args.chapter_seconds * fps))
    quality_metrics = (quality or {}).get("metrics", {})
    if numerical_superiority:
        overload_points = [
            f"平均人数关系 {float(metric(overload, 'mean_attackers', 0)):.1f} 对 {float(metric(overload, 'mean_defenders', 0)):.1f}，峰值优势 {int(metric(overload, 'peak_advantage', 0))} 人。",
            f"持续 {float(metric(overload, 'duration_sec', 0)):.1f} 秒，向前推进 {float(metric(overload, 'forward_progress_m', 0)):.1f} 米。",
            f"进入进攻三区：{'是' if metric(overload, 'entered_final_third', False) else '否'}；8秒内形成射门：{'是' if metric(overload, 'shot_within_8s', False) else '否'}。",
        ]
    else:
        overload_points = [
            f"14 米邻域人数：左队 {metric(overload, 'local_counts', {}).get('left', 0)}，右队 {metric(overload, 'local_counts', {}).get('right', 0)}",
            f"最近对手距离约 {float(metric(overload, 'nearest_opponent_m', 0)):.1f} 米。",
            "人数优势描述空间条件，仍需结合朝向、速度和后续结果判断利用质量。",
        ]
    pressing_reference = metric(pressing, "representative_episode", {})
    pressing_points = [
        f"高位压迫回合 {int(metric(pressing, 'zone_counts', {}).get('high_press', 0))} 个，PPDA视觉代理 {metric(pressing, 'ppda_proxy', '样本不足')}。",
        f"代表回合平均参与 {float(pressing_reference.get('mean_participants', metric(pressing, 'mean_participants', 0))):.1f} 人，最近施压距离 {float(pressing_reference.get('minimum_distance_m', metric(pressing, 'minimum_distance_m', 0))):.1f} 米。",
        "PPDA只表示粗粒度强度；压迫质量还需结合方向、触发、迫使回传和夺回结果。",
    ]
    chapters: list[tuple[float, str, str, list[str], float | None, tuple[int, int, int]]] = [
        (
            2.0,
            (quality or {}).get("title_zh", "输入数据质量评估"),
            (quality or {}).get("summary_zh", "本片段可用于基础候选事件与空间趋势分析。"),
            [
                f"有效分析帧 {quality_metrics.get('observed_frames', 0)}，帧覆盖率 {float(quality_metrics.get('frame_coverage', 0)) * 100:.0f}%",
                f"中位可见球员 {quality_metrics.get('median_visible_players', 0):g} 人，足球帧覆盖率 {float(quality_metrics.get('ball_frame_coverage', 0)) * 100:.0f}%",
                "近景切镜与场地线不足会造成局部坐标缺失。",
            ],
            float((quality or {}).get("confidence", 0.0)),
            GREEN,
        ),
        (
            evidence_time(overload, 2.4),
            (overload or {}).get("title_zh", "局部人数优势候选"),
            (overload or {}).get("summary_zh", "以足球为中心统计双方局部人数关系。"),
            overload_points,
            float((overload or {}).get("confidence", 0.0)),
            CYAN,
        ),
        (
            evidence_time(phase, 8.0),
            (phase or {}).get("title_zh", "比赛阶段分析"),
            (phase or {}).get("summary_zh", "按球权、场区和运动方向聚合有球阶段。"),
            [
                "阶段标签包括后场组织、中场推进、前场进攻与进攻转换。",
                "无球侧同步统计高位、中位和低位防守站位样本。",
                "阶段是可解释规则基线，切镜与比赛中断仍需人工确认。",
            ],
            float((phase or {}).get("confidence", 0.0)) if phase else None,
            GREEN,
        ),
        (
            evidence_time(pass_event, 12.72),
            (pass_event or {}).get("title_zh", "传球候选"),
            (pass_event or {}).get("summary_zh", "持球人切换并伴随足球位移。"),
            [
                f"估计足球位移 {float(metric(pass_event, 'ball_displacement_m', 0)):.1f} 米。",
                "跟踪编号用于连接传球人与接球人，不代表真实球衣号码。",
                "长距离数值容易受足球漏检后重现影响，需要结合画面复核。",
            ],
            float((pass_event or {}).get("confidence", 0.0)),
            CYAN,
        ),
        (
            evidence_time(progression, 18.0),
            (progression or {}).get("title_zh", "推进与穿线分析"),
            (progression or {}).get("summary_zh", "结合传球方向、推进距离和对手站位评估结构突破。"),
            [
                f"向前推进 {float(metric(progression, 'forward_progress_m', 0)):.1f} 米，绕过 {int(metric(progression, 'opponents_bypassed', 0))} 名可见对手。",
                f"跨越 {int(metric(progression, 'lines_broken', 0))} 条防守线代理；区域进入需结合完整镜头复核。",
                "Packing 与穿线是几何候选，不等同于人工标注的成功突破防线事件。",
            ],
            float((progression or {}).get("confidence", 0.0)) if progression else None,
            CYAN,
        ),
        (
            evidence_time(transition, 23.04),
            (transition or {}).get("title_zh", "球权转换候选"),
            (transition or {}).get("summary_zh", "稳定持球方发生切换。"),
            [
                f"5秒向前推进 {float(metric(transition, 'forward_distance_5s_m', 0)):.1f} 米，最多 {int(metric(transition, 'players_joining_attack', 0))} 人高速前插。",
                f"转换速度代理 {float(metric(transition, 'transition_speed_mps', 0)):.2f} m/s，质量代理 {float(metric(transition, 'attacking_transition_quality_proxy', 0)):.0%}。",
                f"回追人数代理 {int(metric(transition, 'recovery_runners', 0))} 人，5秒反抢夺回：{'是' if metric(transition, 'counterpress_regain_within_5s', False) else '否'}。",
            ],
            float((transition or {}).get("confidence", 0.0)) if transition else None,
            YELLOW,
        ),
        (
            evidence_time(pressing, 16.4),
            (pressing or {}).get("title_zh", "压迫分析：当前证据不足"),
            (pressing or {}).get("summary_zh", "需要连续可见持球人与防守球员才能形成压迫回合。"),
            pressing_points,
            float((pressing or {}).get("confidence", 0.0)) if pressing else None,
            YELLOW,
        ),
        (
            evidence_time(shot, max(0.0, duration - 0.4)),
            (shot or {}).get("title_zh", "射门候选"),
            (shot or {}).get("summary_zh", "足球在进攻三区高速朝球门方向运动。"),
            [
                f"估计球速 {float(metric(shot, 'ball_speed_mps', 0)):.2f} m/s，射门距离 {float(metric(shot, 'shot_distance_m', 0)):.2f} m",
                f"球门张角 {float(metric(shot, 'shot_angle_deg', 0)):.2f}°，几何 xG {float(metric(shot, 'xg_lite', 0)):.3f}，上下文代理 {float(metric(shot, 'xg_context_proxy', metric(shot, 'xg_lite', 0))):.3f}",
                f"最近防守人 {float(metric(shot, 'nearest_defender_m', 0) or 0):.1f} 米，射门线封堵 {int(metric(shot, 'blocking_defenders', 0))} 人。",
                "上下文值仍未经过大规模射门结果校准，不可替代正式 xG。",
            ],
            float((shot or {}).get("confidence", 0.0)),
            RED,
        ),
    ]

    if shooting_structure:
        distance_buckets = metric(shooting_structure, "distance_buckets", {})
        pressure_buckets = metric(shooting_structure, "pressure_buckets", {})
        chapters.append(
            (
                evidence_time(shooting_structure, evidence_time(shot, duration * 0.5)),
                shooting_structure.get("title_zh", "球队射门结构"),
                shooting_structure.get("summary_zh", "按位置、距离、通道和防守压力聚合射门候选。"),
                [
                    f"近距离 / 中距离 / 远射：{distance_buckets.get('close', 0)} / {distance_buckets.get('medium', 0)} / {distance_buckets.get('long', 0)} 次。",
                    f"低压 / 中压 / 高压射门：{pressure_buckets.get('low', 0)} / {pressure_buckets.get('medium', 0)} / {pressure_buckets.get('high', 0)} 次。",
                    f"上下文机会质量代理累计 {float(metric(shooting_structure, 'total_xg_context_proxy', 0)):.2f}，不代表正式 xG。",
                ],
                float(shooting_structure.get("confidence", 0.0)),
                RED,
            )
        )
    if set_piece:
        zone_name = {
            "penalty_area": "禁区",
            "out_of_bounds": "场外区域",
            "attacking_half_space": "进攻肋部",
            "wide_channel": "边路",
            "central_channel": "中路",
        }.get(str(metric(set_piece, "landing_zone", "")), "未确认区域")
        chapters.append(
            (
                evidence_time(set_piece, duration * 0.35),
                set_piece.get("title_zh", "定位球落点候选"),
                set_piece.get("summary_zh", "从静止开出追踪至首次稳定落点。"),
                [
                    f"开球速度约 {float(metric(set_piece, 'restart_speed_mps', 0)):.1f} m/s，飞行距离约 {float(metric(set_piece, 'flight_distance_m', 0)):.1f} 米。",
                    f"落点区域：{zone_name}；落点球员 ID：{metric(set_piece, 'landing_track_id', '未确认')}。",
                    f"开球队保持控制：{'是' if metric(set_piece, 'retained_by_taking_team', False) else '否或未确认'}。",
                ],
                float(set_piece.get("confidence", 0.0)),
                YELLOW,
            )
        )
    if goalkeeper:
        chapters.append(
            (
                evidence_time(goalkeeper, evidence_time(shot, duration * 0.5)),
                goalkeeper.get("title_zh", "门将扑救干预候选"),
                goalkeeper.get("summary_zh", "联合射门、门将移动和后续球权判断。"),
                [
                    f"门将与球最近约 {float(metric(goalkeeper, 'minimum_ball_distance_m', 0) or 0):.1f} 米，侧向移动 {float(metric(goalkeeper, 'lateral_displacement_m', 0)):.1f} 米。",
                    f"向场内出击位移代理 {float(metric(goalkeeper, 'outfield_displacement_m', 0)):.1f} 米。",
                    f"射门后防守方形成控制：{'是' if metric(goalkeeper, 'defending_team_controlled_after_shot', False) else '未确认'}；动作类型仍需近景姿态复核。",
                ],
                float(goalkeeper.get("confidence", 0.0)),
                GREEN,
            )
        )
    if attack_chain:
        chapters.append(
            (
                evidence_time(attack_chain, evidence_time(shot, duration * 0.6)),
                attack_chain.get("title_zh", "进攻链路候选"),
                attack_chain.get("summary_zh", "从射门向前回溯同队可见动作。"),
                [
                    f"链路持续 {float(metric(attack_chain, 'duration_sec', 0)):.1f} 秒，共 {int(metric(attack_chain, 'action_count', 0))} 个可见动作。",
                    f"传球 {int(metric(attack_chain, 'pass_count', 0))} 次，推进传球 {int(metric(attack_chain, 'progressive_pass_count', 0))} 次，持球推进 {int(metric(attack_chain, 'carry_count', 0))} 次。",
                    f"累计向前推进约 {float(metric(attack_chain, 'total_forward_progress_m', 0)):.1f} 米，链路受切镜和足球漏检影响。",
                ],
                float(attack_chain.get("confidence", 0.0)),
                CYAN,
            )
        )
    if defensive_risk:
        channel = metric(defensive_risk, "highest_risk_channel", "central")
        channel_name = {"upper_flank": "上方边路", "central": "中路", "lower_flank": "下方边路"}.get(channel, str(channel))
        risk_scores = metric(defensive_risk, "channel_risk_scores", {})
        chapters.append(
            (
                evidence_time(defensive_risk, duration * 0.7),
                defensive_risk.get("title_zh", "防守三区风险"),
                defensive_risk.get("summary_zh", "聚合对手危险进入、穿线、传中、推进和射门。"),
                [
                    f"最高风险通道：{channel_name}，累计风险分 {float(risk_scores.get(channel, 0)):.1f}。",
                    f"可见风险事件 {int(metric(defensive_risk, 'risk_events', 0))} 次。",
                    "风险分只用于同场排序和素材定位，不代表失球概率或个人责任。",
                ],
                float(defensive_risk.get("confidence", 0.0)),
                RED,
            )
        )
    if defensive_gap:
        channel_name = {
            "upper_flank": "上方边路",
            "lower_flank": "下方边路",
            "half_space": "肋部",
            "central": "中路",
        }.get(str(metric(defensive_gap, "channel", "")), "未确认通道")
        chapters.append(
            (
                evidence_time(defensive_gap, duration * 0.75),
                defensive_gap.get("title_zh", "防守空档候选"),
                defensive_gap.get("summary_zh", "联合三线间距和球周防守密度识别可利用空间。"),
                [
                    f"空档通道：{channel_name}；最大线间距约 {float(metric(defensive_gap, 'maximum_line_gap_m', 0)):.1f} 米。",
                    f"球周最近防守人约 {float(metric(defensive_gap, 'maximum_nearest_defender_m', 0)):.1f} 米。",
                    f"8秒内被形成射门：{'是' if metric(defensive_gap, 'shot_within_8s', False) else '否'}；不进行个人失位责任归因。",
                ],
                float(defensive_gap.get("confidence", 0.0)),
                RED,
            )
        )

    insertions: list[tuple[float, np.ndarray]] = []
    for chapter_index, (time_sec, title, summary, points, confidence, accent) in enumerate(chapters, start=1):
        still = still_at(min(time_sec, max(0.0, duration - 1 / fps)))
        rendered = render_chapter(still, chapter_index, title, summary, points, confidence, time_sec, accent)
        if args.interleave_chapters:
            insertions.append((time_sec, rendered))
        else:
            write_repeat(writer, rendered, chapter_frames)

    if not args.interleave_chapters:
        replay_start = max(0.0, evidence_time(shot, duration) - 5.8)
        replay_end = min(duration, evidence_time(shot, duration) + 0.2)
        cap.set(cv2.CAP_PROP_POS_MSEC, replay_start * 1000.0)
        replay_total = max(0.001, replay_end - replay_start)
        replay_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source_time = replay_start + replay_index / fps
            if source_time > replay_end:
                break
            rendered = render_slow_motion(frame, source_time - replay_start, replay_total)
            writer.write(rendered)
            writer.write(rendered)
            replay_index += 1

    left_shape = next((item for item in shapes if "左队" in item.get("title_zh", "")), None)
    right_shape = next((item for item in shapes if "右队" in item.get("title_zh", "")), None)
    shape_frame = still_at(min(15.0, duration / 2))
    shape_points = [
        (left_shape or {}).get("summary_zh", "蓝队缺少足够的完整站位样本。"),
        (right_shape or {}).get("summary_zh", "白队缺少足够的完整站位样本。"),
        "宽度、纵深、三线高度与凸包面积均按有球/无球状态分别聚合，切镜缺失帧不强制补值。",
    ]
    shape_time = evidence_time(left_shape or right_shape, 15.0)
    following_chapter = len(chapters) + 1
    shape_rendered = render_chapter(
        still_at(min(shape_time, duration), shape_frame),
        following_chapter,
        "双方 Team Shape 对比",
        "对照有球与无球状态下的宽度、纵深、三线线距和紧凑度。",
        shape_points,
        0.9,
        shape_time,
        GREEN,
    )
    if args.interleave_chapters:
        insertions.append((shape_time, shape_rendered))
    else:
        write_repeat(writer, shape_rendered, chapter_frames)

    network_points = [
        f"候选网络包含 {int(metric(pass_network, 'nodes', 0))} 个节点、{int(metric(pass_network, 'directed_edges', 0))} 条有向连接。",
        f"归一化传球熵 {float(metric(pass_network, 'normalized_pass_entropy', 0)):.2f}，最大连通分量占比 {float(metric(pass_network, 'largest_component_ratio', 0)):.0%}。",
        "跟踪 ID 不是球衣号码；短片段网络不代表整场组织结构。",
    ]
    network_time = evidence_time(pass_network, duration * 0.72)
    network_rendered = render_chapter(
        still_at(min(network_time, duration), shape_frame),
        following_chapter + 1,
        (pass_network or {}).get("title_zh", "传球网络：当前样本不足"),
        (pass_network or {}).get("summary_zh", "需要更多已确认传球才能形成稳定关系网络。"),
        network_points,
        float((pass_network or {}).get("confidence", 0.0)) if pass_network else None,
        network_time,
        CYAN,
    )
    if args.interleave_chapters:
        insertions.append((network_time, network_rendered))
    else:
        write_repeat(writer, network_rendered, chapter_frames)

    formation_metrics = (formation or {}).get("metrics", {})
    representative = formation_metrics.get("representative_frame", {})
    formation_points = [
        (
            f"有效远景采样 {formation_metrics.get('valid_samples', 0)} 个，其中 "
            f"{formation_metrics.get('supporting_samples', 0)} 个支持主导阵型。"
        ),
        f"代表帧可见 {representative.get('visible_players', 0)} 人，线型 {representative.get('scaled_counts', [])}。",
        "阵型是跨帧可见站位倾向；近景、遮挡和局部镜头不参与强制判型。",
    ]
    formation_time = evidence_time(formation, 15.0)
    formation_still = still_at(min(formation_time, duration), shape_frame)
    formation_rendered = render_formation_chapter(formation_still, report, formation, following_chapter + 2)
    if args.interleave_chapters:
        insertions.append((formation_time, formation_rendered))
    else:
        write_repeat(writer, formation_rendered, chapter_frames)

    if args.interleave_chapters:
        pending = sorted(insertions, key=lambda item: item[0])
        insertion_index = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for index in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            source_time = index / fps
            writer.write(render_playback(frame, args.title, source_time, duration))
            while insertion_index < len(pending) and pending[insertion_index][0] <= source_time:
                write_repeat(writer, pending[insertion_index][1], chapter_frames)
                insertion_index += 1
        while insertion_index < len(pending):
            write_repeat(writer, pending[insertion_index][1], chapter_frames)
            insertion_index += 1

    summary_points = [
        "P0 已覆盖事件时间轴、自动高光、射门结构、区域统计和比赛摘要。",
        "P1 已覆盖平均站位、阵型、Team Shape、定位球落点和门将干预候选。",
        "P2 已覆盖传球类型、持球推进、进攻链路和防守三区风险，并保留证据边界。",
        "P3 已加入动态多打少、高位逼抢/PPDA、防守空档和攻防转换速度候选。",
        (formation or {}).get("summary_zh", "阵型证据不足时不强制输出具体阵型。"),
    ]
    summary_frame = still_at(min(evidence_time(shot, duration), max(0.0, duration - 1 / fps)))
    write_repeat(writer, render_summary(summary_frame, args.title, summary_points), round(args.summary_seconds * fps))

    writer.release()
    cap.release()
    if still_cap is not cap:
        still_cap.release()
    print(f"Output video: {args.output_video}")


if __name__ == "__main__":
    main()
