#!/usr/bin/env python3
"""Create an offensive-technique analysis visualization from SoccerNetGSR JSON."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np

from make_tactical_visualization_video import (
    COLORS,
    HALF_LENGTH,
    HALF_WIDTH,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PROJECT_ROOT,
    Detection,
    blend_rect,
    current_ball,
    draw_mini_pitch,
    load_by_frame,
    pitch_to_panel,
    put_text,
    role_color,
)


GOAL_HALF_WIDTH = 3.66


@dataclass
class Possessor:
    track_id: int
    team: str
    distance: float
    x: float
    y: float


@dataclass
class BallSample:
    frame: int
    x: float
    y: float


class OffensiveState:
    def __init__(self, fps: float):
        self.fps = fps
        self.ball_history: deque[BallSample] = deque(maxlen=20)
        self.latest_possessor: Possessor | None = None
        self.last_touch: Possessor | None = None
        self.last_touch_ball: BallSample | None = None
        self.last_touch_frame = 0

        self.shots = 0
        self.shots_on_target = 0
        self.total_xg = 0.0
        self.last_shot_frame = -9999
        self.latest_shot = "none"
        self.latest_shot_line: tuple[tuple[float, float], tuple[float, float]] | None = None

        self.pass_attempts = 0
        self.pass_completed = 0
        self.pass_types = Counter()
        self.latest_pass = "none"
        self.latest_pass_line: tuple[tuple[float, float], tuple[float, float]] | None = None

        self.control_frames = 0
        self.ball_visible_frames = 0
        self.carries = 0
        self.carry_progress = 0.0
        self.takeon_attempts = 0
        self.takeon_success = 0
        self.carry_start: tuple[int, str, float, float, float, bool] | None = None
        self.latest_carry = "none"

        self.stationary_frames = 0
        self.pending_set_piece: dict | None = None
        self.set_pieces = Counter()
        self.good_set_pieces = 0
        self.latest_set_piece = "none"

    def update(self, frame_idx: int, detections: list[Detection]) -> None:
        ball = current_ball(detections)
        possessor = nearest_possessor(detections, ball)
        self.latest_possessor = possessor

        if ball is None or ball.pitch_x is None or ball.pitch_y is None:
            self._update_carry(frame_idx, None, None, 0.0, detections)
            return

        self.ball_visible_frames += 1
        sample = BallSample(frame_idx, float(ball.pitch_x), float(ball.pitch_y))
        speed = self._ball_speed(sample)
        self.ball_history.append(sample)

        if possessor is not None:
            self.control_frames += 1

        self._detect_set_piece_start(sample, speed)
        self._detect_pass(frame_idx, sample, possessor)
        self._detect_shot(frame_idx, sample, possessor, speed)
        self._update_carry(frame_idx, sample, possessor, speed, detections)
        self._resolve_set_piece(sample, possessor)

    def _ball_speed(self, sample: BallSample) -> float:
        if not self.ball_history:
            return 0.0
        prev = self.ball_history[-1]
        dt = max(1, sample.frame - prev.frame)
        return float(math.hypot(sample.x - prev.x, sample.y - prev.y) * self.fps / dt)

    def _detect_pass(self, frame_idx: int, ball: BallSample, possessor: Possessor | None) -> None:
        if possessor is None:
            return
        if self.last_touch is None:
            self.last_touch = possessor
            self.last_touch_ball = ball
            self.last_touch_frame = frame_idx
            return

        if possessor.track_id == self.last_touch.track_id:
            return

        if self.last_touch_ball is None:
            self.last_touch = possessor
            self.last_touch_ball = ball
            self.last_touch_frame = frame_idx
            return

        gap = frame_idx - self.last_touch_frame
        length = float(math.hypot(ball.x - self.last_touch_ball.x, ball.y - self.last_touch_ball.y))
        if gap < 6 or length < 5.0:
            return

        completed = possessor.team == self.last_touch.team
        pass_type = classify_pass(self.last_touch_ball, ball)
        self.pass_attempts += 1
        self.pass_types[pass_type] += 1
        if completed:
            self.pass_completed += 1
            result = "成功"
        else:
            result = "丢失"
        self.latest_pass = f"{result} {zh_pass_type(pass_type)} {length:.1f}米"
        self.latest_pass_line = ((self.last_touch_ball.x, self.last_touch_ball.y), (ball.x, ball.y))

        self.last_touch = possessor
        self.last_touch_ball = ball
        self.last_touch_frame = frame_idx

    def _detect_shot(self, frame_idx: int, ball: BallSample, possessor: Possessor | None, speed: float) -> None:
        if len(self.ball_history) < 2 or frame_idx - self.last_shot_frame < int(self.fps * 1.2):
            return
        prev = self.ball_history[-2]
        if speed < 18.0 or abs(prev.x) < HALF_LENGTH / 3:
            return

        goal_x = HALF_LENGTH if prev.x > 0 else -HALF_LENGTH
        vx = ball.x - prev.x
        vy = ball.y - prev.y
        to_goal = np.array([goal_x - prev.x, -prev.y], dtype=np.float32)
        vel = np.array([vx, vy], dtype=np.float32)
        if np.linalg.norm(vel) <= 1e-6 or np.linalg.norm(to_goal) <= 1e-6:
            return
        cos_angle = float(np.dot(vel, to_goal) / (np.linalg.norm(vel) * np.linalg.norm(to_goal)))
        if cos_angle < math.cos(math.radians(42)):
            return

        angle, distance = shot_angle_distance(prev.x, prev.y, goal_x)
        xg = xg_lite(distance, angle, prev.x, prev.y)
        on_target = shot_on_target(prev.x, prev.y, vx, vy, goal_x)
        self.shots += 1
        self.total_xg += xg
        if on_target:
            self.shots_on_target += 1
        power = power_label(speed)
        posture = "姿态置信度低" if possessor is not None else "姿态未知"
        self.latest_shot = f"{power} 距离{distance:.1f}米 角度{math.degrees(angle):.0f}度 xG {xg:.2f} {posture}"
        self.latest_shot_line = ((prev.x, prev.y), (goal_x, 0.0))
        self.last_shot_frame = frame_idx

    def _update_carry(
        self,
        frame_idx: int,
        ball: BallSample | None,
        possessor: Possessor | None,
        speed: float,
        detections: list[Detection],
    ) -> None:
        if ball is None or possessor is None:
            self.carry_start = None
            return
        if self.carry_start is None or self.carry_start[1] != possessor.team:
            nearest_def = nearest_defender_distance((ball.x, ball.y), possessor.team, detections)
            attempted = nearest_def <= 6.0
            if attempted:
                self.takeon_attempts += 1
            self.carry_start = (frame_idx, possessor.team, ball.x, ball.y, nearest_def, attempted)
            return
        start_frame, team, sx, sy, start_def_dist, attempted = self.carry_start
        duration = frame_idx - start_frame
        progress = abs(ball.x) - abs(sx)
        if duration >= int(self.fps * 0.8) and progress >= 5.0:
            self.carries += 1
            self.carry_progress += progress
            current_def_dist = nearest_defender_distance((ball.x, ball.y), possessor.team, detections)
            if attempted and current_def_dist - start_def_dist >= 2.5:
                self.takeon_success += 1
            self.latest_carry = f"{zh_team(team)} 推进{progress:.1f}米 速度{speed:.1f}米/秒"
            nearest_def = nearest_defender_distance((ball.x, ball.y), possessor.team, detections)
            attempted = nearest_def <= 6.0
            if attempted:
                self.takeon_attempts += 1
            self.carry_start = (frame_idx, team, ball.x, ball.y, nearest_def, attempted)

    def _detect_set_piece_start(self, ball: BallSample, speed: float) -> None:
        if speed < 1.0:
            self.stationary_frames += 1
            return
        if self.stationary_frames >= int(self.fps * 0.7):
            sp_type = set_piece_type(ball.x, ball.y)
            if sp_type:
                self.pending_set_piece = {
                    "type": sp_type,
                    "start": (ball.x, ball.y),
                    "frames": 0,
                }
                self.set_pieces[sp_type] += 1
                self.latest_set_piece = f"{zh_set_piece(sp_type)} 发起"
        self.stationary_frames = 0

    def _resolve_set_piece(self, ball: BallSample, possessor: Possessor | None) -> None:
        if self.pending_set_piece is None:
            return
        self.pending_set_piece["frames"] += 1
        sx, sy = self.pending_set_piece["start"]
        traveled = float(math.hypot(ball.x - sx, ball.y - sy))
        if traveled < 8.0 and self.pending_set_piece["frames"] < int(self.fps * 3):
            return
        in_box = abs(ball.x) >= HALF_LENGTH - 16.5 and abs(ball.y) <= 20.16
        good = in_box or possessor is not None
        if good:
            self.good_set_pieces += 1
            quality = "落点较好"
        else:
            quality = "落点不明确"
        self.latest_set_piece = f"{zh_set_piece(self.pending_set_piece['type'])} {quality}"
        self.pending_set_piece = None

    def summary(self) -> dict:
        pass_accuracy = self.pass_completed / self.pass_attempts if self.pass_attempts else 0.0
        shot_rate = self.shots_on_target / self.shots if self.shots else 0.0
        stability = self.control_frames / self.ball_visible_frames if self.ball_visible_frames else 0.0
        return {
            "shots": self.shots,
            "shots_on_target": self.shots_on_target,
            "shot_on_target_rate": shot_rate,
            "xg_lite": self.total_xg,
            "passes": self.pass_attempts,
            "completed_passes": self.pass_completed,
            "pass_accuracy": pass_accuracy,
            "pass_types": dict(self.pass_types),
            "carries": self.carries,
            "carry_progress_m": self.carry_progress,
            "control_stability": stability,
            "takeon_attempts": self.takeon_attempts,
            "takeon_success": self.takeon_success,
            "set_pieces": dict(self.set_pieces),
            "good_set_pieces": self.good_set_pieces,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an offensive-analysis mp4 from SoccerNetGSR JSON.")
    parser.add_argument("--input-video", default="soccer_input_dataset/test_5min.mp4")
    parser.add_argument("--json-path", default="soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-1003/SNGS-1003.json")
    parser.add_argument("--output-video", default="soccer_input_dataset/outputs/test_offensive_analysis.mp4")
    parser.add_argument("--snapshot-frame", type=int, default=300)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=900)
    return parser.parse_args()


def nearest_possessor(detections: list[Detection], ball: Detection | None, max_dist: float = 8.0) -> Possessor | None:
    if ball is None or ball.pitch_x is None or ball.pitch_y is None:
        return None
    best: Possessor | None = None
    for det in detections:
        if det.team not in {"left", "right"} or det.role not in {"player", "goalkeeper"}:
            continue
        if det.pitch_x is None or det.pitch_y is None:
            continue
        dist = float(math.hypot(float(det.pitch_x) - float(ball.pitch_x), float(det.pitch_y) - float(ball.pitch_y)))
        if dist <= max_dist and (best is None or dist < best.distance):
            best = Possessor(det.track_id, det.team, dist, float(det.pitch_x), float(det.pitch_y))
    return best


def classify_pass(start: BallSample, end: BallSample) -> str:
    length = float(math.hypot(end.x - start.x, end.y - start.y))
    forward_gain = abs(end.x) - abs(start.x)
    if forward_gain > 10.0 and length >= 12.0:
        return "through"
    if length >= 30.0:
        return "long"
    if abs(end.x - start.x) < 6.0 and abs(end.y - start.y) > 8.0:
        return "lateral"
    if forward_gain < -8.0:
        return "back"
    return "short"


def zh_team(team: str | None) -> str:
    return {"left": "蓝队", "right": "白队"}.get(str(team), str(team))


def zh_pass_type(pass_type: str) -> str:
    return {
        "short": "短传",
        "long": "长传",
        "through": "直塞",
        "lateral": "横传",
        "back": "回传",
    }.get(pass_type, pass_type)


def zh_set_piece(sp_type: str) -> str:
    return {"corner": "角球", "free-kick": "任意球"}.get(sp_type, sp_type)


def offensive_verdict(summary: dict) -> tuple[str, list[str]]:
    shots = int(summary["shots"])
    xg = float(summary["xg_lite"])
    pass_accuracy = float(summary["pass_accuracy"])
    carries = int(summary["carries"])
    carry_progress = float(summary["carry_progress_m"])
    takeon_attempts = int(summary.get("takeon_attempts", 0))
    takeon_success = int(summary.get("takeon_success", 0))

    if shots >= 2 or xg >= 0.35:
        title = "进攻威胁较强：已经形成射门机会"
    elif carries >= 1 and carry_progress >= 6:
        title = "进攻推进有效：能把球带入更深区域"
    elif pass_accuracy >= 0.55:
        title = "组织较稳定：传球连续性较好"
    else:
        title = "进攻效率一般：需要提高连续配合和终结质量"

    notes = [
        f"射门：{shots}次，射正率 {summary['shot_on_target_rate'] * 100:.0f}%，累计 xG {xg:.2f}",
        f"传球：{summary['completed_passes']}/{summary['passes']}，成功率 {pass_accuracy * 100:.0f}%",
        f"推进：{carries}次有效带球，累计向前 {carry_progress:.1f}米，过人 {takeon_success}/{takeon_attempts}",
    ]
    if not summary["set_pieces"]:
        notes.append("定位球：本片段未检测到角球或任意球")
    else:
        notes.append(f"定位球：好落点 {summary['good_set_pieces']} 次")
    return title, notes


def shot_angle_distance(x: float, y: float, goal_x: float) -> tuple[float, float]:
    upper = np.array([goal_x - x, GOAL_HALF_WIDTH - y], dtype=np.float32)
    lower = np.array([goal_x - x, -GOAL_HALF_WIDTH - y], dtype=np.float32)
    denom = max(float(np.linalg.norm(upper) * np.linalg.norm(lower)), 1e-6)
    cos_angle = float(np.dot(upper, lower) / denom)
    angle = math.acos(max(-1.0, min(1.0, cos_angle)))
    distance = float(math.hypot(goal_x - x, y))
    return angle, distance


def xg_lite(distance: float, angle: float, x: float, y: float) -> float:
    in_box = abs(x) >= HALF_LENGTH - 16.5 and abs(y) <= 20.16
    angle_term = min(1.0, angle / math.radians(55))
    distance_term = math.exp(-distance / 18.0)
    box_bonus = 0.12 if in_box else 0.0
    return float(max(0.01, min(0.75, 0.02 + 0.35 * angle_term + 0.28 * distance_term + box_bonus)))


def shot_on_target(x: float, y: float, vx: float, vy: float, goal_x: float) -> bool:
    if abs(vx) < 1e-6:
        return False
    t = (goal_x - x) / vx
    if t <= 0:
        return False
    y_at_goal = y + vy * t
    return -GOAL_HALF_WIDTH <= y_at_goal <= GOAL_HALF_WIDTH


def power_label(speed: float) -> str:
    if speed < 18:
        return "低力量"
    if speed < 28:
        return "可控力量"
    return "高力量"


def set_piece_type(x: float, y: float) -> str | None:
    if abs(x) >= HALF_LENGTH - 6.0 and abs(y) >= HALF_WIDTH - 6.0:
        return "corner"
    if abs(x) >= HALF_LENGTH / 3:
        return "free-kick"
    return None


def nearest_defender_distance(center: tuple[float, float], poss_team: str, detections: list[Detection]) -> float:
    opponent = "right" if poss_team == "left" else "left"
    dists = [
        float(math.hypot(float(det.pitch_x) - center[0], float(det.pitch_y) - center[1]))
        for det in detections
        if det.team == opponent and det.pitch_x is not None and det.pitch_y is not None
    ]
    return min(dists) if dists else 99.0


def draw_event_line(
    frame: np.ndarray,
    line: tuple[tuple[float, float], tuple[float, float]] | None,
    origin: tuple[int, int],
    size: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    if line is None:
        return
    p1 = pitch_to_panel(line[0][0], line[0][1], origin, size)
    p2 = pitch_to_panel(line[1][0], line[1][1], origin, size)
    cv2.arrowedLine(frame, p1, p2, color, 2, cv2.LINE_AA, tipLength=0.18)


def draw_offensive_panel(frame: np.ndarray, frame_idx: int, fps: float, state: OffensiveState) -> None:
    summary = state.summary()
    title, notes = offensive_verdict(summary)
    x1, y1, x2, y2 = 20, frame.shape[0] - 262, 705, frame.shape[0] - 18
    blend_rect(frame, x1, y1, x2, y2, COLORS["panel"], 0.72)
    put_text(frame, "本段进攻评价", (x1 + 14, y1 + 30), scale=0.66, thickness=2)
    put_text(frame, title, (x1 + 14, y1 + 64), scale=0.56, thickness=1)
    put_text(frame, f"时间 {frame_idx / fps:05.2f}s  画面帧 {frame_idx}", (x1 + 14, y1 + 94), scale=0.44)
    for idx, note in enumerate(notes[:4]):
        put_text(frame, note, (x1 + 14, y1 + 124 + idx * 26), scale=0.44)
    pass_types = summary["pass_types"]
    put_text(frame, f"传球结构：短传 {pass_types.get('short',0)}  长传 {pass_types.get('long',0)}  直塞 {pass_types.get('through',0)}  横传 {pass_types.get('lateral',0)}  回传 {pass_types.get('back',0)}", (x1 + 14, y1 + 230), scale=0.42)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    for det in detections:
        if not det.bbox:
            continue
        x = int(det.bbox.get("x", 0))
        y = int(det.bbox.get("y", 0))
        w = int(det.bbox.get("w", 0))
        h = int(det.bbox.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        color = role_color(det)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2 if det.role != "ball" else 3)
        if det.role == "ball":
            label = "球"
        elif det.team in {"left", "right"}:
            label = f"{zh_team(det.team)} {det.track_id}"
        elif det.role == "referee":
            label = f"裁判 {det.track_id}"
        else:
            label = f"ID {det.track_id}"
        put_text(frame, label, (x, max(16, y - 6)), scale=0.42, color=color)


def main() -> int:
    args = parse_args()
    input_video = (PROJECT_ROOT / args.input_video).resolve()
    json_path = (PROJECT_ROOT / args.json_path).resolve()
    output_video = (PROJECT_ROOT / args.output_video).resolve()
    by_frame = load_by_frame(json_path)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(1, args.start_frame)
    if start_frame > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    remaining_frames = max(0, total_frames - start_frame + 1)
    frame_limit = remaining_frames if args.max_frames <= 0 else min(args.max_frames, remaining_frames)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")

    snapshot_path = output_video.with_name(f"{output_video.stem}_frame{args.snapshot_frame:03d}.jpg")
    report_path = output_video.with_suffix(".offensive_report.json")
    state = OffensiveState(fps)
    ball_trail: deque[tuple[float, float]] = deque(maxlen=80)
    local_frame_idx = 1
    wrote_snapshot = False
    while local_frame_idx <= frame_limit:
        ok, frame = cap.read()
        if not ok:
            break
        source_frame_idx = start_frame + local_frame_idx - 1
        detections = by_frame.get(source_frame_idx)
        analysis_frame_idx = source_frame_idx
        if detections is None:
            detections = by_frame.get(local_frame_idx, [])
            analysis_frame_idx = local_frame_idx
        ball = current_ball(detections)
        if ball is not None and ball.pitch_x is not None and ball.pitch_y is not None:
            ball_trail.append((float(ball.pitch_x), float(ball.pitch_y)))
        state.update(analysis_frame_idx, detections)

        out = frame.copy()
        draw_detections(out, detections)
        pitch_w = max(280, out.shape[1] // 4)
        pitch_h = int(pitch_w * PITCH_WIDTH / PITCH_LENGTH)
        origin = (out.shape[1] - pitch_w - 22, 46)
        draw_mini_pitch(out, origin, (pitch_w, pitch_h), detections, ball_trail, None)
        draw_event_line(out, state.latest_pass_line, origin, (pitch_w, pitch_h), (255, 220, 120))
        draw_event_line(out, state.latest_shot_line, origin, (pitch_w, pitch_h), COLORS["danger"])
        draw_offensive_panel(out, source_frame_idx, fps, state)
        writer.write(out)
        if local_frame_idx == args.snapshot_frame:
            cv2.imwrite(str(snapshot_path), out)
            wrote_snapshot = True
        local_frame_idx += 1

    cap.release()
    writer.release()
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(state.summary(), f, ensure_ascii=False, indent=2)
    written = local_frame_idx - 1
    print(f"Wrote {written} frames at {fps:.2f} fps to {output_video}")
    print(f"Wrote offensive report to {report_path}")
    if wrote_snapshot:
        print(f"Wrote snapshot to {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
