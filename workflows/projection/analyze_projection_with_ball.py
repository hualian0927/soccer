"""Fuse the two-camera projection with provided football/player positions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from workflows.projection.analyze_full_pitch_projection import create_report
from tactical_analysis.projection_l1 import L1_GROUPS, analyze_projection_l1


def load_positions(archive: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        required = {"weizhi/ball_positions.json", "weizhi/player_positions.json"}
        if not required.issubset(names):
            raise ValueError(f"Position archive lacks {sorted(required - names)}")
        ball = json.loads(source.read("weizhi/ball_positions.json"))
        players = json.loads(source.read("weizhi/player_positions.json"))
    return ball, players


def boundary_side(x: float, y: float, width: float, length: float, margin: float = 0.8) -> str | None:
    """Return the line crossed by an observed ground projection, if clear of calibration noise."""
    touch_distance = max(-x, x - width)
    goal_distance = max(-y, y - length)
    if touch_distance < margin and goal_distance < margin:
        return None
    return "touchline" if touch_distance >= goal_distance else "goal_line"


def outside_candidates(positions: list[dict], width: float, length: float, fps: float) -> list[dict]:
    """Contiguous, detector-observed outside points are stoppage evidence, not restart proof."""
    candidates = []
    run: list[dict] = []
    for item in sorted(positions, key=lambda row: int(row["frame"])) + [{"frame": -1}]:
        side = None
        if item.get("observed") and float(item.get("confidence") or 0) >= 0.4:
            side = boundary_side(float(item["x_m"]), float(item["y_m"]), width, length)
        if run and (side != run[-1]["side"] or int(item["frame"]) != run[-1]["frame"] + 1):
            if len(run) >= 3:
                first, last = run[0], run[-1]
                candidates.append({
                    "start_sec": round((first["frame"] - 1) / fps, 2),
                    "end_sec": round((last["frame"] - 1) / fps, 2),
                    "line": first["side"],
                    "label_zh": "边线出界迹象" if first["side"] == "touchline" else "底线出界迹象",
                    "possible_restart_zh": "界外球候选" if first["side"] == "touchline" else "角球/球门球候选",
                    "status": "needs_restart_review",
                    "observed_frames": len(run),
                })
            run = []
        if side:
            run.append({"frame": int(item["frame"]), "side": side})
    return candidates


def center_restart_candidates(positions: list[dict], width: float, length: float,
                              duration: float) -> list[dict]:
    """Find a sustained center-spot pause followed by a stable departure."""
    samples = sorted(
        (row for row in positions if row.get("observed") and float(row.get("confidence") or 0) >= 0.4),
        key=lambda row: int(row["frame"]),
    )
    center = np.array([width / 2, length / 2], dtype=float)
    candidates = []
    for index in range(25, len(samples) - 3):
        row = samples[index]
        time_sec = float(row["time_s"])
        if candidates and time_sec - candidates[-1]["timestamp_sec"] < 15:
            continue
        previous = [item for item in samples[max(0, index - 35):index]
                    if time_sec - 1.4 <= float(item["time_s"]) < time_sec]
        if len(previous) < 15:
            continue
        before = np.array([[float(item["x_m"]), float(item["y_m"])] for item in previous])
        if np.max(np.linalg.norm(before - center, axis=1)) > 1.2:
            continue
        future = samples[index:index + 3]
        if any(int(future[offset + 1]["frame"]) != int(future[offset]["frame"]) + 1 for offset in range(2)):
            continue
        after = np.array([[float(item["x_m"]), float(item["y_m"])] for item in future])
        if np.min(np.linalg.norm(after - center, axis=1)) < 2.0:
            continue
        candidates.append({
            "timestamp_sec": round(time_sec, 2),
            "evidence_start_sec": round(max(0, time_sec - 7), 2),
            "evidence_end_sec": round(min(duration, time_sec + 10), 2),
            "type": "kickoff_candidate",
            "label_zh": "中圈开球候选",
            "status": "needs_visual_confirmation",
            "evidence_zh": "中圈附近至少约 1 秒稳定停球，随后连续观测到足球离开中心点。",
        })
    return candidates


def player_index(people: list[dict]) -> dict[int, list[tuple[str, float, float]]]:
    by_frame: dict[int, list[tuple[str, float, float]]] = defaultdict(list)
    for person in people:
        if person.get("role") not in {"Player", "Goalkeeper"} or person.get("color") not in {"red", "blue"}:
            continue
        for position in person.get("positions", []):
            by_frame[int(position["frame"])].append((person["color"], float(position["x_m"]), float(position["y_m"])))
    return by_frame


def ball_window_metrics(ball: list[dict], people: dict[int, list], start: float, end: float,
                        width: float, length: float) -> dict:
    selected = [row for row in ball if start <= float(row["time_s"]) < end and row.get("observed")]
    zones = Counter()
    near = Counter()
    for row in selected:
        x, y = float(row["x_m"]), float(row["y_m"])
        if not (0 <= x <= width and 0 <= y <= length):
            continue
        longitudinal = min(2, int(y / (length / 3)))
        lateral = min(2, int(x / (width / 3)))
        zones[f"{longitudinal + 1}-{lateral + 1}"] += 1
        by_color: dict[str, float] = {}
        for color, px, py in people.get(int(row["frame"]), []):
            distance = math.hypot(x - px, y - py)
            by_color[color] = min(by_color.get(color, float("inf")), distance)
        if by_color:
            closest = min(by_color, key=by_color.get)
            near[closest if by_color[closest] <= 3.0 else "unassigned"] += 1
        else:
            near["unassigned"] += 1
    count = len(selected)
    top_zone = zones.most_common(1)[0][0] if zones else None
    return {
        "observed_ball_frames": count,
        "ball_coverage_ratio": round(count / max(1, int(round((end - start) * 25))), 3),
        "mean_ball_x_m": round(float(np.mean([r["x_m"] for r in selected])), 2) if selected else None,
        "mean_ball_y_m": round(float(np.mean([r["y_m"] for r in selected])), 2) if selected else None,
        "top_zone": top_zone,
        "zone_counts": dict(sorted(zones.items())),
        "nearest_team_within_3m_frames": dict(near),
    }


def build_report(video: Path, archive: Path, geometry: Path, csv_path: Path, window_sec: int) -> dict:
    base = create_report(csv_path, geometry, video, window_sec)
    ball_data, player_data = load_positions(archive)
    pitch = base["pitch"]
    width, length = float(pitch["width_m"]), float(pitch["length_m"])
    fps = float(ball_data["meta"]["fps"])
    capture = cv2.VideoCapture(str(video))
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if abs(video_fps - fps) > 0.01 or video_frames != int(ball_data["meta"]["frames_range"][-1]):
        raise ValueError("Video and ball coordinate timeline do not match")
    people = player_index(player_data["people"])
    positions = ball_data["positions"]
    for window in base["windows"]:
        window["ball"] = ball_window_metrics(positions, people, window["start_sec"], window["end_sec"], width, length)
        window["ball_finding_zh"] = (
            f"足球检出 {window['ball']['observed_ball_frames']} 帧，"
            f"主要位于球场网格 {window['ball']['top_zone']}；近球关系只是距离候选，不能直接判定控球。"
            if window["ball"]["top_zone"] else "此时段足球观测不足，不判断球权或事件。"
        )
    outside = outside_candidates(positions, width, length, fps)
    restarts = center_restart_candidates(positions, width, length, base["summary"]["duration_sec"])
    observed = sum(bool(item.get("observed")) for item in positions)
    base["schema_version"] = "spatial-pitch-ball-v1"
    base["source"]["ball_and_player_positions_zip"] = str(archive.resolve())
    base["summary"].update({
        "ball_observed_frames": observed,
        "ball_coverage_ratio": round(observed / video_frames, 3),
        "out_of_bounds_evidence_count": len(outside),
        "set_piece_restart_candidates": len(restarts),
        "set_piece_restarts_confirmed": 0,
    })
    base["outside_evidence"] = outside
    base["restart_candidates"] = restarts
    base["l1"] = analyze_projection_l1(positions, player_data["people"], restarts, outside,
                                       float(base["summary"]["duration_sec"]), fps)
    base["summary"]["l1_candidate_events"] = base["l1"]["summary"]["total_candidates"]
    for index, restart in enumerate(restarts, 1):
        base["highlights"].append({
            "id": f"center-restart-{index}",
            "start_sec": restart["evidence_start_sec"],
            "end_sec": restart["evidence_end_sec"],
            "title_zh": restart["label_zh"],
            "summary_zh": restart["evidence_zh"] + "具体开球人与结果待原画面复核。",
            "status": restart["status"],
        })
    base["limitations_zh"] = [
        "提供了足球球场投影坐标，但球离地飞行时的二维落点不等于真实落地点；越界需连续检测并回看原画面。",
        "出界只说明可能发生比赛中断；角球、界外球和球门球的类型及开出时间必须由最后触球与重启动作确认。",
        "近球球队仅按 3 米距离给候选，受单应误差、球高度和遮挡影响，不是正式控球率。",
        "队员和球轨迹约有 1-2 米投影误差；双机位拼接与 track_id 切换会影响个人行为统计。",
        "当前球门宽度标定异常，不计算射门角度、进球和 xG。",
    ]
    return base


def markdown_report(report: dict) -> str:
    summary = report["summary"]
    rows = [
        "# 双机位足球与队形技战术分析",
        "",
        f"- 素材：{report['source']['video']}",
        f"- 时长：{summary['duration_sec']:.1f} 秒；逐帧球检测覆盖：{summary['ball_observed_frames']} 帧 ({summary['ball_coverage_ratio']:.1%})。",
        f"- 连续越界证据：{summary['out_of_bounds_evidence_count']} 段；中圈开球候选：{summary['set_piece_restart_candidates']} 段；已确认：0。",
        "- 结论类型：空间站位与近球关系候选；未经人工复核，不作进球、射门或传球官方统计。",
        "",
        "## 时段证据",
        "",
        "| 时间 | 球主要区域 | 近球蓝队帧 | 近球红队帧 | 球观测帧 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for window in report["windows"]:
        ball = window["ball"]
        near = ball["nearest_team_within_3m_frames"]
        rows.append(f"| {window['start_sec']:.0f}-{window['end_sec']:.0f}s | {ball['top_zone'] or '未知'} | "
                    f"{near.get('blue', 0)} | {near.get('red', 0)} | {ball['observed_ball_frames']} |")
    rows.extend(["", "## 出界与定位球", ""])
    if not report["outside_evidence"]:
        rows.append("该 60 秒轨迹没有连续、可信的球场外投影点，不能据此统计界外球、角球或球门球。")
    else:
        for item in report["outside_evidence"]:
            rows.append(f"- {item['start_sec']:.2f}-{item['end_sec']:.2f}s：{item['label_zh']}，"
                        f"{item['possible_restart_zh']}；须复核真实重启。")
    for item in report["restart_candidates"]:
        rows.append(f"- {item['timestamp_sec']:.2f}s：{item['label_zh']}；"
                    f"证据窗 {item['evidence_start_sec']:.2f}-{item['evidence_end_sec']:.2f}s。{item['evidence_zh']}")
    rows.extend(["", "## L1 事件时间轴", "",
                 f"共 {report['l1']['summary']['total_candidates']} 个候选，确认事件 0。近球距离不等于实际触球。", "",
                 "| 时间 | 事件组 | 候选 | 证据说明 |", "| --- | --- | --- | --- |"])
    for event in report["l1"]["events"]:
        rows.append(f"| {event['timestamp_sec']:.2f}s | {event['l1_group']} | {event['label_zh']} | {event['summary_zh']} |")
    rows.extend(["", "### L1 分类覆盖", ""])
    rows.extend(f"- {group} {label}：{report['l1']['summary']['group_counts'][group]} 个候选"
                for group, label in L1_GROUPS.items())
    rows.extend(["", "### 本片不作确认的 L1 项", ""])
    rows.extend(f"- {group}: {reason}" for group, reason in report["l1"]["unsupported_zh"].items()
                if group.startswith("L1-"))
    rows.extend(["", "### 方法限制", "", f"- {report['l1']['unsupported_zh']['cross_camera']}"])
    rows.extend(["", "## 证据边界", ""])
    rows.extend(f"- {warning}" for warning in report["limitations_zh"])
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("二维分析/分析视频2.mp4"))
    parser.add_argument("--positions", type=Path, default=Path("二维分析/weizhi.zip"))
    parser.add_argument("--geometry", type=Path, default=Path("二维分析/full_pitch_field_geometry.json"))
    parser.add_argument("--players-csv", type=Path, default=Path("二维分析/full_pitch_player_field_tracks.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("二维分析/分析视频2_技战术结果"))
    parser.add_argument("--window-sec", type=int, default=5)
    args = parser.parse_args()
    if not 2 <= args.window_sec <= 20:
        parser.error("--window-sec must be between 2 and 20")
    report = build_report(args.video, args.positions, args.geometry, args.players_csv, args.window_sec)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tactical_analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "tactical_analysis_report.md").write_text(markdown_report(report), encoding="utf-8")
    with (args.output_dir / "l1_event_timeline.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        columns = ("event_id", "l1_group", "event_type", "event_subtype", "label_zh", "timestamp_sec",
                   "team_id", "actor_track_id", "recipient_track_id", "confidence", "review_status",
                   "evidence_start_sec", "evidence_end_sec", "summary_zh")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: event.get(key) for key in columns} for event in report["l1"]["events"])
    if report["restart_candidates"]:
        capture = cv2.VideoCapture(str(args.video))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            for index, candidate in enumerate(report["restart_candidates"], 1):
                start = int(candidate["evidence_start_sec"] * fps)
                end = int(candidate["evidence_end_sec"] * fps)
                clip_path = args.output_dir / f"{index:02d}_中圈开球候选.mp4"
                writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot write {clip_path}")
                try:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                    for _ in range(start, end):
                        ok, frame = capture.read()
                        if not ok:
                            break
                        writer.write(frame)
                finally:
                    writer.release()
        finally:
            capture.release()
    print(f"Wrote {args.output_dir}; ball coverage {report['summary']['ball_coverage_ratio']:.1%}; "
          f"outside evidence {report['summary']['out_of_bounds_evidence_count']}; "
          f"restart candidates {report['summary']['set_piece_restart_candidates']}; "
          f"L1 candidates {report['summary']['l1_candidate_events']}")


if __name__ == "__main__":
    main()
