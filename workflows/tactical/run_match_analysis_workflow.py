#!/usr/bin/env python3
"""Scan a full match video and route far/close shots to tactical-analysis modules."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ScanSample:
    frame: int
    time_sec: float
    view: str
    event_score: float
    close_score: float
    far_score: float
    motion_score: float
    scene_change: float
    person_count: int
    ball_count: int
    max_person_height_ratio: float
    mean_person_height_ratio: float
    reason: str


@dataclass
class Segment:
    id: str
    kind: str
    start_sec: float
    end_sec: float
    score: float
    sample_count: int
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-match far/close tactical analysis workflow.")
    parser.add_argument("--input-video", required=True, help="Full match mp4 path.")
    parser.add_argument("--output-root", default="soccer_input_dataset/outputs/match_workflow", help="Directory for reports and rendered clips.")
    parser.add_argument("--mode", choices=["scan", "render"], default="scan", help="scan only or scan plus render selected segments.")
    parser.add_argument("--scan-step-sec", type=float, default=1.0, help="Seconds between sampled frames during coarse scan.")
    parser.add_argument("--clip-padding-sec", type=float, default=4.0, help="Extra context before/after each selected segment.")
    parser.add_argument("--min-segment-sec", type=float, default=5.0)
    parser.add_argument("--max-segment-sec", type=float, default=24.0)
    parser.add_argument("--render-top-close", type=int, default=3, help="Number of close-shot candidates to render in render mode.")
    parser.add_argument("--render-top-far", type=int, default=2, help="Number of far-tactical candidates to render in render mode.")
    parser.add_argument("--scan-yolo-model", default="checkpoints/yolov8n.pt")
    parser.add_argument("--scan-imgsz", type=int, default=640)
    parser.add_argument("--scan-conf", type=float, default=0.25)
    parser.add_argument("--subject-uniform", choices=["auto", "dark", "light", "red", "blue", "yellow"], default="auto")
    parser.add_argument("--team0-colors", default="")
    parser.add_argument("--team1-colors", default="")
    parser.add_argument("--referee-colors", default="")
    parser.add_argument("--goalkeeper-team0-colors", default="")
    parser.add_argument("--goalkeeper-team1-colors", default="")
    parser.add_argument("--fast-gsr", action="store_true", default=True, help="Use lightweight GSR settings for rendered far clips.")
    parser.add_argument("--no-fast-gsr", action="store_false", dest="fast_gsr")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_yolo(model_name: str):
    try:
        from ultralytics import YOLO

        return YOLO(model_name)
    except Exception as exc:
        print(f"[scan] YOLO disabled: {exc}")
        return None


def frame_hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def frame_motion(prev_gray: np.ndarray | None, frame: np.ndarray) -> tuple[float, np.ndarray]:
    gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return 0.0, gray
    diff = cv2.absdiff(gray, prev_gray)
    return float(diff.mean()), gray


def detect_lightweight(model, frame: np.ndarray, imgsz: int, conf: float) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    if model is None:
        return [], []
    result = model.predict(frame, imgsz=imgsz, conf=conf, classes=[0, 32], verbose=False)[0]
    people: list[tuple[float, float, float, float]] = []
    balls: list[tuple[float, float, float, float]] = []
    if result.boxes is None:
        return people, balls
    frame_area = float(frame.shape[0] * frame.shape[1])
    for box, cls_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(float, box)
        if int(cls_id) == 0:
            people.append((x1, y1, x2, y2))
        elif int(cls_id) == 32:
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            if 12.0 <= w * h <= frame_area * 0.006 and 0.45 <= w / h <= 2.2:
                balls.append((x1, y1, x2, y2))
    return people, balls


def classify_sample(
    frame_idx: int,
    fps: float,
    frame: np.ndarray,
    people: list[tuple[float, float, float, float]],
    balls: list[tuple[float, float, float, float]],
    motion: float,
    scene_change: float,
) -> ScanSample:
    height = max(1, frame.shape[0])
    person_heights = [(box[3] - box[1]) / height for box in people]
    max_hr = max(person_heights) if person_heights else 0.0
    mean_hr = float(np.mean(person_heights)) if person_heights else 0.0
    person_count = len(people)
    ball_count = len(balls)

    close_score = 0.0
    close_score += min(1.0, max_hr / 0.58) * 0.52
    close_score += max(0.0, min(1.0, (4.5 - person_count) / 4.5)) * 0.24
    close_score += min(1.0, motion / 28.0) * 0.14
    close_score += (0.10 if ball_count else 0.0)

    far_score = 0.0
    far_score += min(1.0, person_count / 12.0) * 0.48
    far_score += max(0.0, min(1.0, (0.38 - max_hr) / 0.38)) * 0.32
    far_score += min(1.0, motion / 18.0) * 0.10
    far_score += (0.10 if ball_count else 0.0)

    if close_score >= far_score + 0.12:
        view = "close"
    elif far_score >= close_score + 0.08:
        view = "far"
    else:
        view = "mixed"

    event_score = 0.40 * min(1.0, motion / 24.0) + 0.25 * min(1.0, scene_change / 0.65)
    event_score += 0.20 if ball_count else 0.0
    event_score += 0.15 * max(close_score, far_score)
    event_score = float(max(0.0, min(1.0, event_score)))

    reason = f"{view} 人数{person_count} 最大人高{max_hr:.2f} 球{ball_count} 运动{motion:.1f} 切镜{scene_change:.2f}"
    return ScanSample(
        frame=frame_idx,
        time_sec=frame_idx / fps,
        view=view,
        event_score=event_score,
        close_score=float(close_score),
        far_score=float(far_score),
        motion_score=float(motion),
        scene_change=float(scene_change),
        person_count=person_count,
        ball_count=ball_count,
        max_person_height_ratio=float(max_hr),
        mean_person_height_ratio=float(mean_hr),
        reason=reason,
    )


def scan_video(input_video: Path, output_root: Path, args: argparse.Namespace) -> tuple[dict, list[Segment]]:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0.0
    step_frames = max(1, int(args.scan_step_sec * fps))
    model = load_yolo(args.scan_yolo_model)

    samples: list[ScanSample] = []
    prev_hist = None
    prev_gray = None
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        hist = frame_hist(frame)
        scene_change = 0.0 if prev_hist is None else float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        motion, prev_gray = frame_motion(prev_gray, frame)
        prev_hist = hist
        people, balls = detect_lightweight(model, frame, args.scan_imgsz, args.scan_conf)
        samples.append(classify_sample(frame_idx + 1, fps, frame, people, balls, motion, scene_change))
        frame_idx += step_frames
    cap.release()

    segments = build_segments(samples, duration, args)
    manifest = {
        "input_video": str(input_video),
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration,
        "scan_step_sec": args.scan_step_sec,
        "workflow": {
            "far": "远景片段 -> GSR检测/跟踪/队伍角色 -> 战术雷达/阵型/压迫/攻防转换/进攻指标",
            "close": "近景候选 -> YOLO-Pose个人射门分析 -> 分级关键帧暂停 -> 发力链/肢体变化解释",
            "mixed": "混合镜头先作为候选保留，后续人工或二次扫描决定走远景或近景",
        },
        "samples": [asdict(sample) for sample in samples],
        "segments": [asdict(segment) for segment in segments],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "match_workflow_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    write_markdown_report(output_root / "match_workflow_report.md", manifest, segments)
    print(f"[scan] Wrote manifest: {manifest_path}")
    print(f"[scan] Wrote report: {output_root / 'match_workflow_report.md'}")
    return manifest, segments


def build_segments(samples: list[ScanSample], duration: float, args: argparse.Namespace) -> list[Segment]:
    if not samples:
        return []
    raw: list[tuple[str, float, float, float, list[str]]] = []
    active_kind = None
    active_start = 0.0
    active_scores: list[float] = []
    active_reasons: list[str] = []
    step = max(0.1, args.scan_step_sec)

    for sample in samples:
        kind = sample_kind(sample)
        if kind is None:
            if active_kind is not None:
                raw.append((active_kind, active_start, sample.time_sec, float(np.mean(active_scores)), active_reasons[:3]))
            active_kind = None
            active_scores = []
            active_reasons = []
            continue
        if active_kind is None:
            active_kind = kind
            active_start = sample.time_sec
        elif kind != active_kind or sample.time_sec - (active_start + len(active_scores) * step) > step * 2.2:
            raw.append((active_kind, active_start, sample.time_sec, float(np.mean(active_scores)), active_reasons[:3]))
            active_kind = kind
            active_start = sample.time_sec
            active_scores = []
            active_reasons = []
        active_scores.append(sample.event_score)
        active_reasons.append(sample.reason)

    if active_kind is not None:
        raw.append((active_kind, active_start, min(duration, samples[-1].time_sec + step), float(np.mean(active_scores)), active_reasons[:3]))

    merged: list[tuple[str, float, float, float, list[str]]] = []
    for kind, start, end, score, reasons in raw:
        start = max(0.0, start - args.clip_padding_sec)
        end = min(duration, end + args.clip_padding_sec)
        if end - start < args.min_segment_sec:
            center = (start + end) / 2.0
            start = max(0.0, center - args.min_segment_sec / 2.0)
            end = min(duration, start + args.min_segment_sec)
        if end - start > args.max_segment_sec:
            center = (start + end) / 2.0
            start = max(0.0, center - args.max_segment_sec / 2.0)
            end = min(duration, start + args.max_segment_sec)
        if merged and merged[-1][0] == kind and start <= merged[-1][2] + 2.0:
            prev = merged[-1]
            merged[-1] = (kind, prev[1], max(prev[2], end), max(prev[3], score), (prev[4] + reasons)[:4])
        else:
            merged.append((kind, start, end, score, reasons))

    segments = []
    counters = {"close-shot": 0, "far-tactical": 0, "mixed-candidate": 0}
    for kind, start, end, score, reasons in merged:
        counters[kind] += 1
        segments.append(
            Segment(
                id=f"{kind}-{counters[kind]:02d}",
                kind=kind,
                start_sec=round(start, 2),
                end_sec=round(end, 2),
                score=round(score, 3),
                sample_count=max(1, int(math.ceil((end - start) / step))),
                reason="; ".join(reasons),
            )
        )
    return sorted(segments, key=lambda item: (kind_priority(item.kind), -item.score, item.start_sec))


def sample_kind(sample: ScanSample) -> str | None:
    if sample.view == "close" and (sample.event_score >= 0.38 or sample.ball_count > 0):
        return "close-shot"
    if sample.view == "far" and sample.event_score >= 0.30:
        return "far-tactical"
    if sample.view == "mixed" and sample.event_score >= 0.42:
        return "mixed-candidate"
    return None


def kind_priority(kind: str) -> int:
    return {"close-shot": 0, "far-tactical": 1, "mixed-candidate": 2}.get(kind, 9)


def write_markdown_report(path: Path, manifest: dict, segments: list[Segment]) -> None:
    lines = [
        "# 完整赛事远近景技战术分析工作流",
        "",
        f"- 输入视频：`{manifest['input_video']}`",
        f"- 时长：{manifest['duration_sec']:.1f}s",
        f"- 扫描间隔：{manifest['scan_step_sec']}s",
        "",
        "## 工作流",
        "",
        "1. 低频扫描完整比赛，估计镜头尺度、运动强度、切镜和足球可见性。",
        "2. 远景片段进入 GSR/战术模块，输出阵型、压迫、宽度、攻防转换和进攻指标。",
        "3. 近景片段进入个人射门模块，输出姿态、发力链、连续帧变化和高价值关键帧暂停。",
        "4. 混合片段先保留为候选，后续可二次分析或人工审核。",
        "",
        "## 候选片段",
        "",
    ]
    if not segments:
        lines.append("暂无候选片段。可以降低 `--scan-conf` 或 `--scan-step-sec` 再扫。")
    for segment in segments:
        lines.extend(
            [
                f"### {segment.id}",
                "",
                f"- 类型：{zh_kind(segment.kind)}",
                f"- 时间：{format_time(segment.start_sec)} - {format_time(segment.end_sec)}",
                f"- 评分：{segment.score:.3f}",
                f"- 原因：{segment.reason}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def zh_kind(kind: str) -> str:
    return {
        "close-shot": "近景个人射门候选",
        "far-tactical": "远景整体战术候选",
        "mixed-candidate": "混合镜头候选",
    }.get(kind, kind)


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def run_command(label: str, command: list[str]) -> None:
    print(f"\n[{label}] {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def render_segments(input_video: Path, output_root: Path, segments: list[Segment], args: argparse.Namespace) -> None:
    close_segments = [s for s in segments if s.kind == "close-shot"][: args.render_top_close]
    far_segments = [s for s in segments if s.kind == "far-tactical"][: args.render_top_far]
    render_root = output_root / "rendered_segments"
    render_root.mkdir(parents=True, exist_ok=True)

    for idx, segment in enumerate(close_segments, 1):
        duration = max(1.0, segment.end_sec - segment.start_sec)
        out_video = render_root / f"{segment.id}_individual.mp4"
        run_command(
            f"close-{idx}-individual",
            [
                sys.executable,
                "-m", "workflows.visualization.make_individual_technique_analysis_video",
                "--input-video",
                str(input_video),
                "--start-sec",
                f"{segment.start_sec:.2f}",
                "--duration-sec",
                f"{duration:.2f}",
                "--infer-step",
                "1",
                "--subject-uniform",
                args.subject_uniform,
                "--keyframe-pause-sec",
                "0.6",
                "--keyframe-pause-policy",
                "high-value",
                "--max-keyframes",
                "4",
                "--output-video",
                str(out_video),
            ],
        )

    for idx, segment in enumerate(far_segments, 1):
        start_frame = max(1, int(segment.start_sec * get_video_fps(input_video)) + 1)
        max_frames = max(1, int((segment.end_sec - segment.start_sec) * get_video_fps(input_video)))
        video_name = f"MATCH-FAR-{idx:03d}"
        gsr_video = render_root / f"{segment.id}_gsr.mp4"
        work_root = output_root / "gsr_work"
        command = [
            sys.executable,
            "-m", "workflows.gsr.run_local_video_gsr_visualization",
            "--input-video",
            str(input_video),
            "--video-name",
            video_name,
            "--start-frame",
            str(start_frame),
            "--max-frames",
            str(max_frames),
            "--output-video",
            str(gsr_video),
            "--work-root",
            str(work_root),
            "--batch-size",
            "4",
            "--recall-optimized",
            "--overwrite",
            "--team0-colors",
            args.team0_colors,
            "--team1-colors",
            args.team1_colors,
            "--referee-colors",
            args.referee_colors,
            "--goalkeeper-team0-colors",
            args.goalkeeper_team0_colors,
            "--goalkeeper-team1-colors",
            args.goalkeeper_team1_colors,
        ]
        if args.fast_gsr:
            command.append("--fast-mode")
        run_command(f"far-{idx}-gsr", command)

        json_path = work_root / "SoccerNetGS" / "test" / video_name / f"{video_name}.json"
        if json_path.exists():
            tactical_video = render_root / f"{segment.id}_tactical.mp4"
            offensive_video = render_root / f"{segment.id}_offensive.mp4"
            run_command(
                f"far-{idx}-tactical",
                [
                    sys.executable,
                    "-m", "workflows.visualization.make_tactical_visualization_video",
                    "--input-video",
                    str(input_video),
                    "--json-path",
                    str(json_path),
                    "--start-frame",
                    str(start_frame),
                    "--max-frames",
                    str(max_frames),
                    "--yolo-fallback",
                    "--fallback-backend",
                    "ultralytics",
                    "--output-video",
                    str(tactical_video),
                ],
            )
            run_command(
                f"far-{idx}-offensive",
                [
                    sys.executable,
                    "-m", "workflows.visualization.make_offensive_analysis_video",
                    "--input-video",
                    str(input_video),
                    "--json-path",
                    str(json_path),
                    "--start-frame",
                    str(start_frame),
                    "--max-frames",
                    str(max_frames),
                    "--output-video",
                    str(offensive_video),
                ],
            )


def get_video_fps(input_video: Path) -> float:
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return float(fps)


def main() -> int:
    args = parse_args()
    input_video = resolve_path(args.input_video)
    output_root = resolve_path(args.output_root)
    _, segments = scan_video(input_video, output_root, args)
    if args.mode == "render":
        render_segments(input_video, output_root, segments, args)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
