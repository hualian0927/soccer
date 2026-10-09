"""Analyze a two-camera pitch projection without inventing ball events."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from review_tactical_candidates_with_vision import call_vision_api


REQUIRED_COLUMNS = {"frame", "time_s", "camera", "track_id", "role", "color", "x_m", "y_m"}
TEAM_COLORS = ("blue", "red")
UNSUPPORTED_ACTION_WORDS = (
    "射门", "传球", "角球", "任意球", "进球", "控球", "逼抢", "扑救", "反击",
    "进攻", "防守", "靠前", "靠后", "推进", "前场", "后场",
)


def deduplicate_players(group: pd.DataFrame, threshold_m: float = 2.2) -> tuple[np.ndarray, int]:
    """Match only cross-camera observations of the same colored player."""
    cameras = sorted(group["camera"].unique())
    if len(cameras) != 2:
        return group[["x_m", "y_m"]].to_numpy(dtype=float), 0
    left = group[group.camera == cameras[0]][["x_m", "y_m"]].to_numpy(dtype=float)
    right = group[group.camera == cameras[1]][["x_m", "y_m"]].to_numpy(dtype=float)
    candidates = sorted(
        (float(np.linalg.norm(a - b)), i, j)
        for i, a in enumerate(left)
        for j, b in enumerate(right)
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    merged = []
    for distance, i, j in candidates:
        if distance > threshold_m:
            break
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            merged.append((left[i] + right[j]) / 2)
    merged.extend(point for i, point in enumerate(left) if i not in used_left)
    merged.extend(point for j, point in enumerate(right) if j not in used_right)
    return np.asarray(merged, dtype=float).reshape(-1, 2), len(used_left)


def shape_metrics(points: np.ndarray, pitch_width: float, pitch_length: float) -> dict | None:
    if len(points) < 5:
        return None
    x = points[:, 0]
    y = points[:, 1]
    hull = cv2.convexHull(points.astype(np.float32))
    return {
        "visible_players": len(points),
        "centroid_x_m": round(float(np.median(x)), 2),
        "centroid_y_m": round(float(np.median(y)), 2),
        "width_m": round(float(np.percentile(x, 90) - np.percentile(x, 10)), 2),
        "depth_m": round(float(np.percentile(y, 90) - np.percentile(y, 10)), 2),
        "convex_area_m2": round(float(cv2.contourArea(hull)), 2),
        "largest_y_gap_m": round(float(np.max(np.diff(np.sort(y)))), 2),
        "third_counts": [int(np.sum((y >= lo) & (y < hi))) for lo, hi in (
            (0, pitch_length / 3), (pitch_length / 3, 2 * pitch_length / 3),
            (2 * pitch_length / 3, pitch_length + 1),
        )],
        "channel_counts": [int(np.sum((x >= lo) & (x < hi))) for lo, hi in (
            (0, pitch_width / 3), (pitch_width / 3, 2 * pitch_width / 3),
            (2 * pitch_width / 3, pitch_width + 1),
        )],
    }


def frame_metrics(data: pd.DataFrame, pitch_width: float, pitch_length: float) -> list[dict]:
    records = []
    players = data[(data.role == "Player") & data.color.isin(TEAM_COLORS)]
    for frame, frame_rows in players.groupby("frame", sort=True):
        record = {"frame": int(frame), "time_sec": round(float(frame_rows.time_s.iloc[0]), 3)}
        positions: dict[str, np.ndarray] = {}
        duplicates = 0
        for color in TEAM_COLORS:
            points, merged = deduplicate_players(frame_rows[frame_rows.color == color])
            positions[color] = points
            duplicates += merged
            record[color] = shape_metrics(points, pitch_width, pitch_length)
        record["cross_camera_duplicates_merged"] = duplicates
        red, blue = positions["red"], positions["blue"]
        if len(red) and len(blue):
            distances = np.linalg.norm(red[:, None, :] - blue[None, :, :], axis=2)
            record["close_opponent_pairs_3m"] = int(np.sum(distances < 3.0))
            record["nearest_opponent_m"] = round(float(np.min(distances)), 2)
        else:
            record["close_opponent_pairs_3m"] = 0
            record["nearest_opponent_m"] = None
        records.append(record)
    return records


def median(values: list[float | int]) -> float | None:
    return round(float(np.median(values)), 2) if values else None


def aggregate_window(records: list[dict], index: int, start: float, end: float) -> dict:
    sample = [record for record in records if start <= record["time_sec"] < end]
    if not sample:
        raise ValueError(f"{start:.1f}-{end:.1f} 秒没有有效球员数据")
    teams = {}
    for color in TEAM_COLORS:
        valid = [record[color] for record in sample if record[color]]
        teams[color] = {
            key: median([value[key] for value in valid])
            for key in (
                "visible_players", "centroid_x_m", "centroid_y_m", "width_m", "depth_m",
                "convex_area_m2", "largest_y_gap_m",
            )
        }
        teams[color]["third_counts"] = [median([item["third_counts"][i] for item in valid]) for i in range(3)]
        teams[color]["channel_counts"] = [median([item["channel_counts"][i] for item in valid]) for i in range(3)]
        teams[color]["valid_frame_ratio"] = round(len(valid) / len(sample), 3)
    middle = min(sample, key=lambda item: abs(item["time_sec"] - (start + end) / 2))
    return {
        "id": f"spatial-{index + 1:02d}",
        "start_sec": round(start, 2),
        "end_sec": round(end, 2),
        "representative_sec": middle["time_sec"],
        "observed_frames": len(sample),
        "teams": teams,
        "median_close_opponent_pairs_3m": median([item["close_opponent_pairs_3m"] for item in sample]),
        "median_nearest_opponent_m": median([
            item["nearest_opponent_m"] for item in sample if item["nearest_opponent_m"] is not None
        ]),
        "median_cross_camera_duplicates_merged": median([
            item["cross_camera_duplicates_merged"] for item in sample
        ]),
    }


def describe_window(window: dict, previous: dict | None) -> str:
    blue, red = window["teams"]["blue"], window["teams"]["red"]
    parts = []
    for name, team in (("蓝队", blue), ("红队", red)):
        parts.append(f"{name}可见球员中位数{team['visible_players']:.0f}人，宽度约{team['width_m']:.1f}米、纵深约{team['depth_m']:.1f}米")
    if previous:
        shifts = []
        for color, name in (("blue", "蓝队"), ("red", "红队")):
            change = window["teams"][color]["centroid_y_m"] - previous["teams"][color]["centroid_y_m"]
            if abs(change) >= 3.0:
                shifts.append(f"{name}重心沿场地纵向{'向右' if change > 0 else '向左'}移动约{abs(change):.1f}米")
        parts.extend(shifts)
    return "；".join(parts) + "。"


def create_report(csv_path: Path, geometry_path: Path, video_path: Path, window_sec: int) -> dict:
    data = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"坐标表缺少字段: {sorted(missing)}")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    width = float(geometry["pitch"]["width_m"])
    length = float(geometry["pitch"]["length_m"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法读取二维视频: {video_path}")
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    records = frame_metrics(data, width, length)
    windows = []
    previous = None
    for index, start in enumerate(np.arange(0, duration, window_sec)):
        window = aggregate_window(records, index, float(start), min(float(start + window_sec), duration))
        window["algorithm_finding_zh"] = describe_window(window, previous)
        windows.append(window)
        previous = window
    summary = {
        "duration_sec": round(duration, 2),
        "source_frames": int(data.frame.nunique()),
        "player_rows": int((data.role == "Player").sum()),
        "camera_count": int(data.camera.nunique()),
        "windows": len(windows),
        "median_cross_camera_duplicates_merged_per_frame": median([
            item["cross_camera_duplicates_merged"] for item in records
        ]),
        "teams": {
            color: {
                key: median([window["teams"][color][key] for window in windows])
                for key in ("visible_players", "width_m", "depth_m", "convex_area_m2", "largest_y_gap_m")
            }
            for color in TEAM_COLORS
        },
        "vision_reviewed_windows": 0,
    }
    opening = windows[: min(5, len(windows))]
    closing = windows[-min(4, len(windows)):]
    first_gap = median([
        abs(item["teams"]["red"]["centroid_y_m"] - item["teams"]["blue"]["centroid_y_m"])
        for item in opening
    ])
    last_gap = median([
        abs(item["teams"]["red"]["centroid_y_m"] - item["teams"]["blue"]["centroid_y_m"])
        for item in closing
    ])
    highlights = [
        {
            "id": "shape-opening",
            "start_sec": opening[0]["start_sec"],
            "end_sec": opening[-1]["end_sec"],
            "title_zh": "前段：蓝队覆盖更宽、更深",
            "summary_zh": "在可见球员中，蓝队的横向宽度和纵向深度中位数均高于红队；这是站位差异，不代表控球或进攻优势。",
            "metrics": {
                color: {
                    key: median([item["teams"][color][key] for item in opening])
                    for key in ("visible_players", "width_m", "depth_m")
                }
                for color in TEAM_COLORS
            },
            "status": "geometry_observation",
        },
        {
            "id": "shape-convergence",
            "start_sec": closing[0]["start_sec"],
            "end_sec": closing[-1]["end_sec"],
            "title_zh": "后段：双方纵向重心靠近",
            "summary_zh": f"双方纵向重心间距由前段约{first_gap:.1f}米缩至后段约{last_gap:.1f}米；仅描述队形位置变化，不能据此判定压迫或攻防转换。",
            "metrics": {"opening_centroid_gap_m": first_gap, "closing_centroid_gap_m": last_gap},
            "status": "geometry_observation",
        },
    ]
    warnings = [
        "缺少足球地面轨迹 CSV 与原始双机位比赛画面：不能确认控球、传球、射门、角球、进球及门将扑救。",
        "team=0/1 为机位内部编号；全局红蓝队以 Player 行的 color 合并，同色跨机位近邻已去重。",
        "track_id 仅在单个机位内有效；不提供跨机位球员身份或个人跑动总距离。",
        "给定球场模型的 goal_width_m=21.2526 异常，本报告不计算射门角度或门前距离。",
        "球场单应与双机位拼接存在位置误差；米制宽度、纵深仅作样本内相对比较。",
        "阵型名称、攻防转换、压迫与局部优势成效依赖球权、比赛方向和完整人员观察，本报告不作确定判定。",
    ]
    return {
        "schema_version": "spatial-pitch-v1",
        "source": {
            "video": str(video_path.resolve()),
            "tracks_csv": str(csv_path.resolve()),
            "geometry_json": str(geometry_path.resolve()),
            "fps": float(geometry["source"]["fps"]),
        },
        "pitch": {"width_m": width, "length_m": length},
        "summary": summary,
        "highlights": highlights,
        "windows": windows,
        "limitations_zh": warnings,
    }


def vision_request(video_path: Path, window: dict, capture: cv2.VideoCapture) -> dict:
    start, end = window["start_sec"], window["end_sec"]
    times = np.linspace(start + 0.1, end - 0.1, 5)
    content = [{
        "type": "text",
        "text": (
            "你是足球空间站位分析的证据复核员。输入是双机位融合后的二维俯视示意图，红蓝点代表队员；"
            "不是原始比赛画面，也没有足球位置或球权数据。按时间顺序比较下面5帧，只解释可见队形的横向宽度、"
            "纵向深度、重心与双方相对距离，不猜测具体阵型、射门、传球、角球、控球或压迫。"
            "数字仅提供上下文，不必重复。若画面无法确认变化，直说不确定。只返回 JSON 对象："
            "summary_zh(不超过60字), observations_zh(最多2条具体视觉证据), "
            "uncertainty_zh(一句限制说明), evidence_frame_indices(0-4整数数组), visual_confidence(0到1)。"
            f"\n算法窗口: {json.dumps({'start_sec': start, 'end_sec': end, 'teams': window['teams']}, ensure_ascii=False)}"
        ),
    }]
    for index, second in enumerate(times):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"无法读取二维视频第 {second:.2f} 秒")
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise ValueError("证据帧编码失败")
        content.extend((
            {"type": "text", "text": f"证据帧 {index}，时间 {second:.2f} 秒"},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
                "detail": "high",
            }},
        ))
    return {
        "model": "deepseek-flash",
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 420,
        "messages": [{"role": "user", "content": content}],
    }


def sanitize_vision(response: dict) -> dict:
    summary = str(response.get("summary_zh") or "").strip()
    observations = response.get("observations_zh")
    if not summary or not isinstance(observations, list):
        raise ValueError("视觉模型未返回有效空间分析 JSON")
    text = summary + " ".join(str(item) for item in observations)
    if any(word in text for word in UNSUPPORTED_ACTION_WORDS):
        return {
            "status": "withheld_unsupported_action_claim",
            "summary_zh": "视觉文本包含当前数据不能支持的事件或方向判断，已隐藏，保留二维量化结果。",
            "observations_zh": [],
            "uncertainty_zh": "缺少足球轨迹和原始比赛画面。",
        }
    return {
        "status": "spatial_interpretation_only",
        "summary_zh": summary[:120],
        "observations_zh": [str(item)[:120] for item in observations[:2]],
        "uncertainty_zh": str(response.get("uncertainty_zh") or "仅能复核可见站位")[:160],
        "evidence_frame_indices": [
            index for index in response.get("evidence_frame_indices", [])
            if isinstance(index, int) and 0 <= index < 5
        ],
        "visual_confidence": response.get("visual_confidence"),
    }


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("二维分析/full_pitch_player_field_tracks.csv"))
    parser.add_argument("--geometry", type=Path, default=Path("二维分析/full_pitch_field_geometry.json"))
    parser.add_argument("--video", type=Path, default=Path("二维分析/分析视频.mp4"))
    parser.add_argument("--output", type=Path, default=Path("二维分析/spatial_analysis_report.json"))
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument("--vision", action="store_true", help="对每个窗口调用 DeepSeek 图片 API")
    args = parser.parse_args()
    if args.window_sec < 2 or args.window_sec > 20:
        parser.error("--window-sec 应在 2-20 秒之间")
    existing = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else None
    report = create_report(args.csv, args.geometry, args.video, args.window_sec)
    if existing and existing.get("source") == report["source"]:
        prior = {window["id"]: window for window in existing.get("windows", [])}
        for window in report["windows"]:
            old = prior.get(window["id"])
            if old and old.get("start_sec") == window["start_sec"] and old.get("end_sec") == window["end_sec"] and old.get("vision"):
                prior_vision = old["vision"]
                clean_vision = (
                    {**prior_vision, "summary_zh": "视觉文本包含当前数据不能支持的事件或方向判断，已隐藏，保留二维量化结果。"}
                    if prior_vision.get("status") == "withheld_unsupported_action_claim"
                    else sanitize_vision(prior_vision)
                )
                clean_vision["model"] = prior_vision.get("model", "deepseek-flash")
                clean_vision["usage"] = prior_vision.get("usage")
                window["vision"] = clean_vision
    report["summary"]["vision_reviewed_windows"] = sum(bool(item.get("vision")) for item in report["windows"])
    write_report(report, args.output)
    if not args.vision:
        print(f"二维空间报告已写入: {args.output}")
        return
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        parser.error("未设置 DEEPSEEK_API_KEY；二维报告已生成，但尚未调用视觉 API")
    capture = cv2.VideoCapture(str(args.video))
    try:
        for window in report["windows"]:
            if window.get("vision"):
                continue
            payload = vision_request(args.video, window, capture)
            response, usage = call_vision_api(payload, key)
            window["vision"] = sanitize_vision(response)
            window["vision"]["model"] = "deepseek-flash"
            window["vision"]["usage"] = usage
            report["summary"]["vision_reviewed_windows"] = sum(
                bool(item.get("vision")) for item in report["windows"]
            )
            write_report(report, args.output)
            print(f"已复核 {window['id']} ({window['start_sec']:.0f}-{window['end_sec']:.0f} 秒)", flush=True)
    finally:
        capture.release()
    report["summary"]["vision_reviewed_windows"] = sum(bool(item.get("vision")) for item in report["windows"])
    write_report(report, args.output)
    print(f"二维 + 视觉报告已写入: {args.output}")


if __name__ == "__main__":
    main()
