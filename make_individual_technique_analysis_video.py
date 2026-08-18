#!/usr/bin/env python3
"""Create a close-up individual technique analysis video with YOLO-Pose."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from make_tactical_visualization_video import COLORS, PROJECT_ROOT, blend_rect, put_text


COCO_KPTS = {
    "nose": 0,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


@dataclass
class PersonPose:
    bbox: tuple[float, float, float, float]
    score: float
    keypoints: np.ndarray
    conf: np.ndarray


@dataclass
class BallBox:
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    score: float


@dataclass
class KeyframeSnapshot:
    frame_idx: int
    image: np.ndarray
    metrics: dict[str, float | str | bool]
    tier: int


def metric_float(metrics: dict[str, float | str | bool], key: str, default: float = 0.0) -> float:
    return float(metrics.get(key, default))


class IndividualTechniqueState:
    def __init__(self, fps: float):
        self.fps = fps
        self.prev_pose: PersonPose | None = None
        self.prev_ball: BallBox | None = None
        self.prev_ball_frame = 0
        self.frame_count = 0
        self.pose_visible_frames = 0
        self.ball_visible_frames = 0
        self.shot_candidate_frames = 0
        self.tackle_risk_frames = 0
        self.left_foot_activity = 0.0
        self.right_foot_activity = 0.0
        self.max_ball_speed = 0.0
        self.max_ankle_speed = 0.0
        self.posture_scores: list[float] = []
        self.latest_notes = ["等待姿态检测"]
        self.latest_verdict = "个人动作分析初始化"
        self.latest_power = "未知"
        self.latest_foot = "未知"
        self.prev_ball_to_foot = 0.0
        self.last_near_ball_frame = -10**9
        self.last_near_ball_dist = 0.0
        self.phase_counts = Counter()
        self.event_history: deque[str] = deque(maxlen=9)
        self.motion_history: deque[dict[str, float | str | bool]] = deque(maxlen=max(6, int(fps * 0.64)))
        self.display_verdict = "个人动作分析初始化"
        self.display_notes = ["等待姿态检测"]

    def update(self, frame_idx: int, pose: PersonPose | None, ball: BallBox | None) -> dict[str, float | str | bool]:
        self.frame_count += 1
        metrics: dict[str, float | str | bool] = {
            "posture_score": 0.0,
            "torso_lean": 0.0,
            "left_knee": 0.0,
            "right_knee": 0.0,
            "left_ankle_speed": 0.0,
            "right_ankle_speed": 0.0,
            "ball_speed": 0.0,
            "ball_to_foot": 0.0,
            "balance_score": 0.0,
            "plant_score": 0.0,
            "swing_score": 0.0,
            "contact_score": 0.0,
            "leg_shape_score": 0.0,
            "leg_shot_score": 0.0,
            "head_to_ball": 0.0,
            "head_speed": 0.0,
            "header_score": 0.0,
            "shot_action_score": 0.0,
            "shot_type": "未判定",
            "goalkeeper_like": False,
            "shot_candidate": False,
            "tackle_risk": False,
            "near_ball": False,
            "release_detected": False,
        }

        if ball is not None:
            self.ball_visible_frames += 1
            if self.prev_ball is not None:
                dt = max(1, frame_idx - self.prev_ball_frame)
                jump = pixel_dist(ball.center, self.prev_ball.center)
                if jump <= 220.0 * dt:
                    speed = jump * self.fps / dt
                    self.max_ball_speed = max(self.max_ball_speed, speed)
                    metrics["ball_speed"] = speed
                    self.latest_power = power_label(speed)
            self.prev_ball = ball
            self.prev_ball_frame = frame_idx

        if pose is None:
            self.latest_verdict = "未稳定检测到主体球员"
            self.latest_notes = ["近景遮挡或镜头切换时，姿态关键点可能丢失"]
            self.event_history.append(self.latest_verdict)
            self.display_verdict = Counter(self.event_history).most_common(1)[0][0]
            if self.latest_verdict == self.display_verdict:
                self.display_notes = self.latest_notes
            return metrics

        self.pose_visible_frames += 1
        torso_lean = torso_lean_deg(pose)
        left_knee = joint_angle(pose, "left_hip", "left_knee", "left_ankle")
        right_knee = joint_angle(pose, "right_hip", "right_knee", "right_ankle")
        left_speed = ankle_speed(pose, self.prev_pose, "left_ankle")
        right_speed = ankle_speed(pose, self.prev_pose, "right_ankle")
        head_speed = point_speed(pose, self.prev_pose, "nose")
        self.left_foot_activity += left_speed
        self.right_foot_activity += right_speed
        self.max_ankle_speed = max(self.max_ankle_speed, left_speed, right_speed)
        self.latest_foot = foot_candidate(self.left_foot_activity, self.right_foot_activity)

        posture_score = shot_posture_score(pose, torso_lean, left_knee, right_knee, left_speed, right_speed)
        components = technique_components(pose, ball, torso_lean, left_knee, right_knee, left_speed, right_speed)
        leg_shape_score = kicking_leg_shape_score(pose, left_knee, right_knee, left_speed, right_speed)
        self.posture_scores.append(posture_score)
        ball_to_foot = float(components["ball_to_foot"])
        foot_speed = max(left_speed, right_speed)
        ball_speed = float(metrics["ball_speed"])
        near_ball = 0.0 < ball_to_foot <= 105.0
        recent_near_ball = frame_idx - self.last_near_ball_frame <= max(2, int(self.fps * 0.32))
        head_to_ball = nearest_head_distance(pose, ball)
        header_score = header_action_score(pose, ball, torso_lean, head_speed)
        goalkeeper_like = is_goalkeeper_save_like_pose(pose, ball, torso_lean)
        if goalkeeper_like:
            header_score = 0.0
        leg_shot_score = leg_shot_action_score(
            posture_score,
            float(components["plant_score"]),
            float(components["swing_score"]),
            float(components["contact_score"]),
            leg_shape_score,
            ball_to_foot,
            foot_speed,
            ball_speed,
            recent_near_ball,
        )
        if goalkeeper_like:
            leg_shot_score *= 0.45
        release_detected = (
            near_ball
            and self.prev_ball_to_foot > 0.0
            and ball_to_foot - self.prev_ball_to_foot >= 32.0
            and ball_speed >= 520.0
        ) or (
            0.0 < ball_to_foot <= 135.0
            and foot_speed >= 48.0
            and ball_speed >= 950.0
        ) or (
            recent_near_ball
            and 0.0 < ball_to_foot <= 240.0
            and foot_speed >= 70.0
            and ball_speed >= 1200.0
        ) or (
            recent_near_ball
            and 0.0 < ball_to_foot <= 560.0
            and foot_speed >= 30.0
            and ball_speed >= 1800.0
            and posture_score >= 75.0
        ) or (
            0.0 < ball_to_foot <= 220.0
            and foot_speed >= 90.0
            and ball_speed >= 1600.0
            and posture_score >= 75.0
        )
        pose_confirmed_leg_shot = leg_shot_score >= 78.0 and (near_ball or recent_near_ball)
        pose_confirmed_header = header_score >= 72.0
        shot_action_score = max(leg_shot_score, header_score)
        shot_type = shot_type_label(leg_shot_score, header_score, release_detected)
        shot_candidate = bool(
            release_detected
            or pose_confirmed_leg_shot
            or pose_confirmed_header
            or (near_ball and foot_speed >= 52.0 and posture_score >= 70.0 and components["contact_score"] >= 38.0)
        )
        tackle_risk = sliding_tackle_proxy(pose, torso_lean)
        phase = action_phase(
            posture_score,
            left_speed,
            right_speed,
            ball_to_foot,
            ball_speed,
            release_detected,
            tackle_risk,
            leg_shot_score,
            header_score,
        )
        self.phase_counts[phase] += 1
        if shot_candidate:
            self.shot_candidate_frames += 1
        if tackle_risk:
            self.tackle_risk_frames += 1

        metrics.update(
            {
                "posture_score": posture_score,
                "torso_lean": torso_lean,
                "left_knee": left_knee,
                "right_knee": right_knee,
                "left_ankle_speed": left_speed,
                "right_ankle_speed": right_speed,
                "head_speed": head_speed,
                "ball_to_foot": ball_to_foot,
                "balance_score": components["balance_score"],
                "plant_score": components["plant_score"],
                "swing_score": components["swing_score"],
                "contact_score": components["contact_score"],
                "leg_shape_score": leg_shape_score,
                "leg_shot_score": leg_shot_score,
                "head_to_ball": head_to_ball,
                "header_score": header_score,
                "shot_action_score": shot_action_score,
                "shot_type": shot_type,
                "goalkeeper_like": goalkeeper_like,
                "phase": phase,
                "shot_candidate": shot_candidate,
                "tackle_risk": tackle_risk,
                "near_ball": near_ball,
                "release_detected": release_detected,
            }
        )
        self.motion_history.append(metrics.copy())
        self.latest_verdict, self.latest_notes = self.make_verdict(metrics)
        self.event_history.append(self.latest_verdict)
        self.display_verdict = Counter(self.event_history).most_common(1)[0][0]
        if self.latest_verdict == self.display_verdict:
            self.display_notes = self.latest_notes
        if near_ball:
            self.last_near_ball_frame = frame_idx
            self.last_near_ball_dist = ball_to_foot
        if ball_to_foot > 0.0:
            self.prev_ball_to_foot = ball_to_foot
        self.prev_pose = pose
        return metrics

    def make_verdict(self, metrics: dict[str, float | str | bool]) -> tuple[str, list[str]]:
        score = float(metrics["posture_score"])
        phase = str(metrics.get("phase", "未知阶段"))
        if bool(metrics["tackle_risk"]):
            title = "疑似倒地/铲抢动作：需结合上下文判断规范性"
        elif phase == "头球攻门候选":
            title = "头球攻门候选：头部接近球路，身体前压发力"
        elif phase == "大幅摆腿/抽射候选":
            title = "抽射/大幅摆腿候选：肢体动作已进入击球窗口"
        elif phase == "持球推进/调整":
            title = "持球推进中：暂不进入射门动作判定"
        elif phase == "射门准备":
            title = "射门准备阶段：等待触球关键帧"
        elif bool(metrics["shot_candidate"]):
            title = "疑似射门动作：摆腿充分，动作进入击球阶段"
        elif score >= 68:
            title = "射门准备姿态较规范"
        elif score >= 48:
            title = "射门姿态基本可用，但稳定性一般"
        else:
            title = "当前画面不适合判定射门姿态"

        notes = self.professional_analysis_lines(metrics, compact=True)
        return title, notes

    def professional_analysis_lines(self, metrics: dict[str, float | str | bool], compact: bool = False) -> list[str]:
        phase = str(metrics.get("phase", "未知阶段"))
        lean = float(metrics.get("torso_lean", 0.0))
        posture = float(metrics.get("posture_score", 0.0))
        plant = float(metrics.get("plant_score", 0.0))
        swing = float(metrics.get("swing_score", 0.0))
        contact = float(metrics.get("contact_score", 0.0))
        left_speed = float(metrics.get("left_ankle_speed", 0.0))
        right_speed = float(metrics.get("right_ankle_speed", 0.0))
        ball_speed = float(metrics.get("ball_speed", 0.0))
        ball_to_foot = float(metrics.get("ball_to_foot", 0.0))
        shot_type = str(metrics.get("shot_type", "未判定"))
        leg_score = float(metrics.get("leg_shot_score", 0.0))
        header_score = float(metrics.get("header_score", 0.0))
        head_to_ball = float(metrics.get("head_to_ball", 0.0))
        kick_side = "左脚" if left_speed > right_speed else "右脚"
        plant_side = "右脚" if kick_side == "左脚" else "左脚"
        ref = self.motion_history[0] if self.motion_history else metrics
        ref_swing_speed = max(float(ref.get("left_ankle_speed", 0.0)), float(ref.get("right_ankle_speed", 0.0)))
        swing_speed = max(left_speed, right_speed)
        swing_gain = swing_speed - ref_swing_speed
        lean_delta = lean - float(ref.get("torso_lean", 0.0))
        current_knee = float(metrics.get("left_knee" if kick_side == "左脚" else "right_knee", 0.0))
        ref_knee = float(ref.get("left_knee" if kick_side == "左脚" else "right_knee", 0.0))
        knee_delta = current_knee - ref_knee
        ref_ball_dist = float(ref.get("ball_to_foot", 0.0))
        ball_release = ball_to_foot - ref_ball_dist if ball_to_foot > 0.0 and ref_ball_dist > 0.0 else 0.0

        shape = shot_body_shape_label(lean, posture, plant, swing)
        chain = force_chain_label(plant_side, kick_side, plant, swing, contact)
        change = (
            f"连续帧变化：摆腿速度{signed_value(swing_gain)}px/f，"
            f"{kick_side}膝角{signed_value(knee_delta)}度，躯干{signed_value(lean_delta)}度"
        )
        release = release_reason_label(phase, ball_speed, ball_to_foot, ball_release, contact)
        muscle = muscle_proxy_label(plant_side, kick_side, phase, swing_gain, knee_delta)
        action_read = action_evidence_label(shot_type, leg_score, header_score, ball_to_foot, head_to_ball, ball_speed)
        scores = f"分项：平衡 {float(metrics.get('balance_score', 0.0)):.0f}  支撑 {plant:.0f}  摆腿 {swing:.0f}  触球 {contact:.0f}  头球 {header_score:.0f}"

        if compact:
            return [
                f"形态：{shape}  阶段：{phase}  姿态 {posture:.0f}/100",
                f"发力链：{chain}",
                action_read,
                muscle,
                release,
            ]
        return [
            f"动作形态：{shape}",
            f"发力链条：{chain}",
            change,
            action_read,
            release,
            muscle,
            scores,
            f"惯用脚倾向：{self.latest_foot}  力量代理：{self.latest_power}",
        ]

    def summary(self) -> dict:
        avg_score = float(np.mean(self.posture_scores)) if self.posture_scores else 0.0
        best_score = float(np.max(self.posture_scores)) if self.posture_scores else 0.0
        return {
            "frames": self.frame_count,
            "pose_visible_frames": self.pose_visible_frames,
            "ball_visible_frames": self.ball_visible_frames,
            "avg_posture_score": avg_score,
            "best_posture_score": best_score,
            "shot_candidate_frames": self.shot_candidate_frames,
            "tackle_risk_frames": self.tackle_risk_frames,
            "phase_counts": dict(self.phase_counts),
            "dominant_foot_candidate": self.latest_foot,
            "left_foot_activity": self.left_foot_activity,
            "right_foot_activity": self.right_foot_activity,
            "max_ankle_speed_px_per_frame": self.max_ankle_speed,
            "max_ball_speed_px_per_sec": self.max_ball_speed,
            "power_proxy": self.latest_power,
            "latest_professional_analysis": self.display_notes,
            "limitations": "基于转播画面、2D人体关键点和像素速度估计；真实惯用脚、射门力量和动作规范性需要更多标注、多角度或相机标定。",
        }


class GoalkeeperTechniqueState:
    def __init__(self, fps: float):
        self.fps = fps
        self.prev_pose: PersonPose | None = None
        self.frame_count = 0
        self.visible_frames = 0
        self.dive_frames = 0
        self.save_attempts = 0
        self.successful_save_proxies = 0
        self.last_dive_active = False
        self.current_attempt_success = False
        self.last_save_frame = -10**9
        self.latest_verdict = "门将防守分析初始化"
        self.latest_notes = ["等待门将姿态检测"]
        self.display_verdict = self.latest_verdict
        self.display_notes = self.latest_notes
        self.event_history: deque[str] = deque(maxlen=8)
        self.shot_seen_frame = -10**9
        self.latest_shot_speed = 0.0
        self.best_reaction_sec: float | None = None
        self.best_save_distance = 1e9

    def update(
        self,
        frame_idx: int,
        pose: PersonPose | None,
        ball: BallBox | None,
        shooter_metrics: dict[str, float | str | bool],
    ) -> dict[str, float | str | bool]:
        self.frame_count += 1
        if is_high_value_shot_frame(shooter_metrics) or is_key_technique_frame(shooter_metrics):
            self.shot_seen_frame = frame_idx
            self.latest_shot_speed = float(shooter_metrics.get("ball_speed", 0.0))

        metrics: dict[str, float | str | bool] = {
            "goalkeeper_visible": False,
            "goalkeeper_phase": "未检测",
            "dive_score": 0.0,
            "set_position_score": 0.0,
            "extension_score": 0.0,
            "save_distance": 0.0,
            "reaction_sec": -1.0,
            "save_attempt": False,
            "save_success_proxy": False,
        }
        if pose is None:
            self.latest_verdict = "未稳定检测到门将"
            self.latest_notes = ["近景切换或遮挡时，门将可能不在画面内"]
            self._smooth_display()
            return metrics

        self.visible_frames += 1
        torso_lean = torso_lean_deg(pose)
        x1, y1, x2, y2 = pose.bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        body_ratio = w / h
        center_speed = 0.0 if self.prev_pose is None else pixel_dist(bbox_center(pose.bbox), bbox_center(self.prev_pose.bbox))
        extension_score = goalkeeper_extension_score(pose)
        set_position_score = goalkeeper_set_position_score(pose, torso_lean, body_ratio)
        dive_score = goalkeeper_dive_score(torso_lean, body_ratio, center_speed, extension_score)
        ball_dist = nearest_hand_or_body_distance(pose, ball)
        reaction_sec = -1.0
        if frame_idx - self.shot_seen_frame <= int(self.fps * 2.0):
            reaction_sec = max(0.0, (frame_idx - self.shot_seen_frame) / max(self.fps, 1.0))

        phase = goalkeeper_phase(dive_score, set_position_score, reaction_sec)
        save_attempt = dive_score >= 62.0 and 0.0 <= reaction_sec <= 1.6
        save_success_proxy = save_attempt and ball_dist > 0.0 and ball_dist <= 145.0
        if dive_score >= 62.0:
            self.dive_frames += 1
        if save_attempt and not self.last_dive_active and frame_idx - self.last_save_frame > int(self.fps * 0.8):
            self.save_attempts += 1
            self.last_save_frame = frame_idx
            self.current_attempt_success = False
            if self.best_reaction_sec is None or reaction_sec < self.best_reaction_sec:
                self.best_reaction_sec = reaction_sec
        if save_success_proxy and not self.current_attempt_success:
            self.successful_save_proxies += 1
            self.current_attempt_success = True
        if ball_dist > 0.0:
            self.best_save_distance = min(self.best_save_distance, ball_dist)
        self.last_dive_active = dive_score >= 62.0

        metrics.update(
            {
                "goalkeeper_visible": True,
                "goalkeeper_phase": phase,
                "dive_score": dive_score,
                "set_position_score": set_position_score,
                "extension_score": extension_score,
                "save_distance": ball_dist,
                "reaction_sec": reaction_sec,
                "save_attempt": save_attempt,
                "save_success_proxy": save_success_proxy,
            }
        )
        self.latest_verdict, self.latest_notes = self.make_verdict(metrics)
        self.event_history.append(self.latest_verdict)
        self._smooth_display()
        self.prev_pose = pose
        return metrics

    def make_verdict(self, metrics: dict[str, float | str | bool]) -> tuple[str, list[str]]:
        phase = str(metrics.get("goalkeeper_phase", "未知"))
        dive = float(metrics.get("dive_score", 0.0))
        set_pos = float(metrics.get("set_position_score", 0.0))
        ext = float(metrics.get("extension_score", 0.0))
        save_dist = float(metrics.get("save_distance", 0.0))
        reaction = float(metrics.get("reaction_sec", -1.0))
        if bool(metrics.get("save_success_proxy")):
            title = "门将扑救代理：疑似接近球路/形成有效封堵"
        elif bool(metrics.get("save_attempt")):
            title = "门将扑救动作：侧扑或倒地扑救已启动"
        elif phase == "准备站位":
            title = "门将准备站位：等待射门触发"
        else:
            title = "门将防守画面：可做姿态观察"
        reaction_text = "未关联射门" if reaction < 0 else f"{reaction:.2f}s"
        distance_text = "未知" if save_dist <= 0.0 else f"{save_dist:.0f}px"
        notes = [
            f"阶段：{phase}  准备 {set_pos:.0f}/100  扑救 {dive:.0f}/100  伸展 {ext:.0f}/100",
            f"出击/反应：射门后 {reaction_text}  手/身体到球距离 {distance_text}",
            goalkeeper_technique_line(phase, dive, set_pos, ext),
            "脚下技术/指挥防线：当前近景未覆盖，需门将持球或防线组织片段",
        ]
        return title, notes

    def _smooth_display(self) -> None:
        if self.event_history:
            self.display_verdict = Counter(self.event_history).most_common(1)[0][0]
            if self.latest_verdict == self.display_verdict:
                self.display_notes = self.latest_notes
        else:
            self.display_verdict = self.latest_verdict
            self.display_notes = self.latest_notes

    def summary(self) -> dict:
        save_rate = self.successful_save_proxies / self.save_attempts if self.save_attempts else 0.0
        return {
            "goalkeeper_visible_frames": self.visible_frames,
            "goalkeeper_dive_frames": self.dive_frames,
            "save_attempts_proxy": self.save_attempts,
            "successful_saves_proxy": self.successful_save_proxies,
            "save_success_rate_proxy": save_rate,
            "best_reaction_sec_proxy": self.best_reaction_sec,
            "best_save_distance_px": None if self.best_save_distance >= 1e9 else self.best_save_distance,
            "latest_goalkeeper_analysis": self.display_notes,
            "limitations": "门将分析基于单目2D姿态、球框和像素距离；真实扑救成功率、出击时机、脚下技术和指挥能力需要更多镜头、事件标注或多机位数据。",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze close-up football technique with YOLO-Pose.")
    parser.add_argument("--input-video", default="soccer_input_dataset/进球近景/videoplayback.mp4")
    parser.add_argument("--output-video", default="soccer_input_dataset/outputs/goal_closeup_individual_technique_zh.mp4")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt")
    parser.add_argument("--ball-model", default="yolov8n.pt")
    parser.add_argument("--start-sec", type=float, default=105.0)
    parser.add_argument("--duration-sec", type=float, default=18.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.35)
    parser.add_argument("--ball-conf", type=float, default=0.18)
    parser.add_argument("--infer-step", type=int, default=2)
    parser.add_argument("--snapshot-frame", type=int, default=180)
    parser.add_argument("--draw-moving-labels", action="store_true", help="Draw labels attached to moving boxes.")
    parser.add_argument("--smooth-alpha", type=float, default=1.0, help="Pose drawing smoothing, lower is smoother.")
    parser.add_argument("--keyframe-pause-sec", type=float, default=0.0, help="Freeze and explain shooting keyframes.")
    parser.add_argument("--keyframe-cooldown-sec", type=float, default=2.0, help="Minimum time between two keyframe pauses.")
    parser.add_argument("--max-keyframes", type=int, default=12, help="Maximum number of keyframe pauses to insert.")
    parser.add_argument("--keyframe-pause-policy", choices=["high-value", "all"], default="high-value", help="Pause only high-value shots or every keyframe candidate.")
    parser.add_argument("--minor-keyframe-pause-sec", type=float, default=0.0, help="Optional short pause for ordinary touch candidates under high-value policy.")
    parser.add_argument("--keyframe-lookback-sec", type=float, default=0.48, help="When ball speed confirms a shot, explain the best pre-contact frame from this lookback window.")
    parser.add_argument("--keyframe-segment-sec", type=float, default=0.44, help="Replay this many seconds of the key shooting action instead of freezing a single frame only.")
    parser.add_argument("--subject-uniform", choices=["auto", "dark", "light", "red", "blue", "yellow", "yellowgreen"], default="auto", help="Optional visual prior for selecting the main player.")
    parser.add_argument("--goalkeeper-uniform", choices=["auto", "dark", "light", "red", "blue", "yellow", "yellowgreen"], default="yellowgreen", help="Optional visual prior for selecting the goalkeeper.")
    parser.add_argument("--disable-goalkeeper-analysis", action="store_true", help="Disable goalkeeper defensive technique analysis.")
    return parser.parse_args()


def pixel_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def keypoint(pose: PersonPose, name: str, min_conf: float = 0.20) -> tuple[float, float] | None:
    idx = COCO_KPTS[name]
    if idx >= len(pose.keypoints) or pose.conf[idx] < min_conf:
        return None
    x, y = pose.keypoints[idx]
    return float(x), float(y)


def joint_angle(pose: PersonPose, a_name: str, b_name: str, c_name: str) -> float:
    a = keypoint(pose, a_name)
    b = keypoint(pose, b_name)
    c = keypoint(pose, c_name)
    if a is None or b is None or c is None:
        return 0.0
    va = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    vc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
    denom = max(float(np.linalg.norm(va) * np.linalg.norm(vc)), 1e-6)
    cos_value = float(np.dot(va, vc) / denom)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos_value)))))


def torso_lean_deg(pose: PersonPose) -> float:
    ls = keypoint(pose, "left_shoulder")
    rs = keypoint(pose, "right_shoulder")
    lh = keypoint(pose, "left_hip")
    rh = keypoint(pose, "right_hip")
    if None in {ls, rs, lh, rh}:
        return 0.0
    shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    return abs(float(math.degrees(math.atan2(dx, -dy))))


def ankle_speed(pose: PersonPose, prev_pose: PersonPose | None, name: str) -> float:
    return point_speed(pose, prev_pose, name)


def point_speed(pose: PersonPose, prev_pose: PersonPose | None, name: str) -> float:
    if prev_pose is None:
        return 0.0
    point = keypoint(pose, name)
    prev = keypoint(prev_pose, name)
    if point is None or prev is None:
        return 0.0
    return pixel_dist(point, prev)


def shot_posture_score(
    pose: PersonPose,
    torso_lean: float,
    left_knee: float,
    right_knee: float,
    left_speed: float,
    right_speed: float,
) -> float:
    visible_names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    visibility = sum(1 for name in visible_names if keypoint(pose, name) is not None) / len(visible_names)
    lean_score = max(0.0, 1.0 - abs(torso_lean - 16.0) / 36.0)
    knee_extension = max(left_knee, right_knee) / 180.0 if max(left_knee, right_knee) > 0 else 0.0
    swing_score = min(1.0, max(left_speed, right_speed) / 30.0)
    bbox_w = pose.bbox[2] - pose.bbox[0]
    bbox_h = max(1.0, pose.bbox[3] - pose.bbox[1])
    upright_score = max(0.0, min(1.0, bbox_h / max(1.0, bbox_w * 1.15)))
    score = 28 * visibility + 24 * lean_score + 24 * knee_extension + 16 * swing_score + 8 * upright_score
    return float(max(0.0, min(100.0, score)))


def technique_components(
    pose: PersonPose,
    ball: BallBox | None,
    torso_lean: float,
    left_knee: float,
    right_knee: float,
    left_speed: float,
    right_speed: float,
) -> dict[str, float]:
    bbox_w = max(1.0, pose.bbox[2] - pose.bbox[0])
    bbox_h = max(1.0, pose.bbox[3] - pose.bbox[1])
    lean_fit = max(0.0, 1.0 - abs(torso_lean - 14.0) / 34.0)
    upright_fit = max(0.0, min(1.0, bbox_h / max(1.0, bbox_w * 1.25)))
    balance_score = 100.0 * (0.55 * lean_fit + 0.45 * upright_fit)

    left_support = support_foot_score(pose, "left", left_speed, left_knee)
    right_support = support_foot_score(pose, "right", right_speed, right_knee)
    plant_score = max(left_support, right_support)

    swing_leg_speed = max(left_speed, right_speed)
    swing_knee = right_knee if right_speed >= left_speed else left_knee
    swing_speed_score = min(100.0, swing_leg_speed / 2.8)
    swing_extension_score = max(0.0, min(100.0, (swing_knee - 80.0) / 90.0 * 100.0)) if swing_knee > 0 else 0.0
    swing_score = 0.62 * swing_speed_score + 0.38 * swing_extension_score

    ball_to_foot = nearest_foot_distance(pose, ball)
    contact_score = 0.0 if ball_to_foot <= 0 else max(0.0, min(100.0, 100.0 - ball_to_foot / 2.2))

    return {
        "balance_score": float(max(0.0, min(100.0, balance_score))),
        "plant_score": float(max(0.0, min(100.0, plant_score))),
        "swing_score": float(max(0.0, min(100.0, swing_score))),
        "contact_score": float(max(0.0, min(100.0, contact_score))),
        "ball_to_foot": float(ball_to_foot),
    }


def support_foot_score(pose: PersonPose, side: str, ankle_speed_value: float, knee_angle: float) -> float:
    ankle = keypoint(pose, f"{side}_ankle")
    hip = keypoint(pose, f"{side}_hip")
    if ankle is None or hip is None:
        return 0.0
    stability = max(0.0, min(1.0, 1.0 - ankle_speed_value / 70.0))
    knee_fit = max(0.0, min(1.0, (knee_angle - 110.0) / 45.0)) if knee_angle > 0 else 0.0
    vertical_support = max(0.0, min(1.0, abs(ankle[1] - hip[1]) / 260.0))
    return float(100.0 * (0.48 * stability + 0.32 * knee_fit + 0.20 * vertical_support))


def nearest_foot_distance(pose: PersonPose, ball: BallBox | None) -> float:
    if ball is None:
        return 0.0
    feet = [keypoint(pose, "left_ankle"), keypoint(pose, "right_ankle")]
    dists = [pixel_dist(point, ball.center) for point in feet if point is not None]
    return min(dists) if dists else 0.0


def nearest_head_distance(pose: PersonPose, ball: BallBox | None) -> float:
    if ball is None:
        return 0.0
    points = [keypoint(pose, "nose"), keypoint(pose, "left_shoulder"), keypoint(pose, "right_shoulder")]
    dists = [pixel_dist(point, ball.center) for point in points if point is not None]
    return min(dists) if dists else 0.0


def kicking_leg_shape_score(
    pose: PersonPose,
    left_knee: float,
    right_knee: float,
    left_speed: float,
    right_speed: float,
) -> float:
    kick_side = "left" if left_speed > right_speed else "right"
    hip = keypoint(pose, f"{kick_side}_hip")
    knee = keypoint(pose, f"{kick_side}_knee")
    ankle = keypoint(pose, f"{kick_side}_ankle")
    if hip is None or knee is None or ankle is None:
        return 0.0
    diag = max(1.0, bbox_diag(pose.bbox))
    knee_angle = left_knee if kick_side == "left" else right_knee
    hip_to_ankle = pixel_dist(hip, ankle) / diag
    ankle_to_knee = pixel_dist(ankle, knee) / diag
    lateral_reach = abs(ankle[0] - hip[0]) / diag
    extension = max(0.0, min(1.0, (knee_angle - 92.0) / 76.0)) if knee_angle > 0 else 0.0
    reach = max(0.0, min(1.0, (hip_to_ankle - 0.22) / 0.42))
    lower_leg = max(0.0, min(1.0, ankle_to_knee / 0.28))
    lateral = max(0.0, min(1.0, lateral_reach / 0.28))
    return float(100.0 * (0.38 * extension + 0.30 * reach + 0.20 * lower_leg + 0.12 * lateral))


def leg_shot_action_score(
    posture: float,
    plant: float,
    swing: float,
    contact: float,
    leg_shape: float,
    ball_to_foot: float,
    foot_speed: float,
    ball_speed: float,
    recent_near_ball: bool,
) -> float:
    if ball_to_foot <= 0.0:
        ball_relation = 0.0
    else:
        ball_relation = max(0.0, min(100.0, 100.0 - abs(ball_to_foot - 58.0) * 0.85))
    motion = min(100.0, foot_speed / 1.15)
    release = min(100.0, ball_speed / 22.0)
    recent_bonus = 8.0 if recent_near_ball else 0.0
    score = (
        0.22 * posture
        + 0.17 * plant
        + 0.20 * swing
        + 0.16 * contact
        + 0.15 * leg_shape
        + 0.07 * ball_relation
        + 0.03 * release
        + recent_bonus
    )
    if motion >= 58.0 and leg_shape >= 55.0:
        score += 10.0
    return float(max(0.0, min(100.0, score)))


def header_action_score(pose: PersonPose, ball: BallBox | None, torso_lean: float, head_speed: float) -> float:
    nose = keypoint(pose, "nose")
    ls = keypoint(pose, "left_shoulder")
    rs = keypoint(pose, "right_shoulder")
    if ball is None or nose is None or ls is None or rs is None:
        return 0.0
    diag = max(1.0, bbox_diag(pose.bbox))
    head_dist = nearest_head_distance(pose, ball)
    ball_above_head = ball.center[1] <= nose[1] + max(20.0, diag * 0.12)
    if not ball_above_head:
        return 0.0
    close = max(0.0, min(100.0, 100.0 - head_dist / max(1.0, diag * 0.0062)))
    shoulder_y = (ls[1] + rs[1]) / 2.0
    head_above = max(0.0, min(100.0, (shoulder_y - nose[1]) / max(1.0, diag * 0.26) * 100.0))
    lean = max(0.0, min(100.0, torso_lean / 0.36))
    head_motion = max(0.0, min(100.0, head_speed / 0.55))
    score = 0.42 * close + 0.22 * head_above + 0.20 * lean + 0.16 * head_motion
    if goalkeeper_dive_score(torso_lean, bbox_width_height_ratio(pose.bbox), 0.0, goalkeeper_extension_score(pose)) >= 56.0:
        score -= 36.0
    if head_dist <= max(75.0, diag * 0.34) and (torso_lean >= 14.0 or head_speed >= 10.0):
        score += 14.0
    return float(max(0.0, min(100.0, score)))


def shot_type_label(leg_score: float, header_score: float, release_detected: bool) -> str:
    if header_score >= 72.0 and header_score >= leg_score + 6.0:
        return "头球候选"
    if leg_score >= 84.0:
        return "大幅摆腿/抽射候选"
    if leg_score >= 72.0:
        return "脚下射门候选"
    if release_detected:
        return "球速释放候选"
    return "未判定"


def action_phase(
    posture_score: float,
    left_speed: float,
    right_speed: float,
    ball_to_foot: float,
    ball_speed: float,
    release_detected: bool,
    tackle_risk: bool,
    leg_shot_score: float = 0.0,
    header_score: float = 0.0,
) -> str:
    if tackle_risk:
        return "倒地/铲抢候选"
    max_speed = max(left_speed, right_speed)
    near_ball = 0.0 < ball_to_foot <= 105.0
    if header_score >= 72.0:
        return "头球攻门候选"
    if release_detected:
        return "触球/击球瞬间"
    if leg_shot_score >= 82.0:
        return "大幅摆腿/抽射候选"
    if near_ball and max_speed >= 62.0 and posture_score >= 66.0:
        return "摆腿加速"
    if near_ball and (max_speed >= 18.0 or ball_speed > 0.0):
        return "持球推进/调整"
    if posture_score >= 72 and max_speed >= 34.0:
        return "射门准备"
    return "观察/跑动"


def technique_advice(balance: float, plant: float, swing: float, contact: float) -> str:
    weakest = min(
        [("身体平衡", balance), ("支撑脚稳定", plant), ("摆腿质量", swing), ("触球时机", contact)],
        key=lambda item: item[1],
    )
    advice = {
        "身体平衡": "建议：降低后仰，保持髋肩轴稳定",
        "支撑脚稳定": "建议：支撑脚落点靠近球侧，减少晃动",
        "摆腿质量": "建议：增加摆腿幅度并完成随摆",
        "触球时机": "建议：在球靠近脚背时完成发力",
    }
    return advice[weakest[0]]


def signed_value(value: float) -> str:
    return f"+{value:.0f}" if value >= 0.0 else f"{value:.0f}"


def shot_body_shape_label(lean: float, posture: float, plant: float, swing: float) -> str:
    if lean >= 38.0:
        shape = "明显前倾压低重心，身体呈前压折线"
    elif lean >= 20.0:
        shape = "髋肩前倾形成弓形发力姿态"
    elif posture >= 78.0 and plant >= 60.0:
        shape = "上体相对稳定，支撑脚建立制动轴"
    else:
        shape = "身体形态仍在调整，发力轴未完全建立"
    if swing >= 72.0:
        return f"{shape}，摆腿进入鞭打加速"
    return f"{shape}，摆腿仍在蓄力"


def force_chain_label(plant_side: str, kick_side: str, plant: float, swing: float, contact: float) -> str:
    plant_text = "支撑脚制动稳定" if plant >= 65.0 else "支撑脚稳定性一般"
    swing_text = "髋带动大腿、小腿鞭打" if swing >= 68.0 else "摆腿幅度仍在建立"
    contact_text = "踝部锁紧完成释放" if contact >= 55.0 else "脚面触球时机仍需观察"
    return f"{plant_side}{plant_text}，{kick_side}{swing_text}，{contact_text}"


def muscle_proxy_label(plant_side: str, kick_side: str, phase: str, swing_gain: float, knee_delta: float) -> str:
    if phase == "触球/击球瞬间":
        return f"肌群代理：{plant_side}臀腿抗旋稳住骨盆，{kick_side}髋屈带动伸膝，踝关节固定触球"
    if swing_gain >= 45.0 or knee_delta >= 18.0:
        return f"肌群代理：{kick_side}大腿先加速，小腿随后伸展，形成近端到远端的鞭打"
    return f"肌群代理：{plant_side}负责支撑制动，{kick_side}仍在蓄力，尚未完成最终释放"


def release_reason_label(phase: str, ball_speed: float, ball_to_foot: float, ball_release: float, contact: float) -> str:
    if phase == "头球攻门候选":
        return "技巧原因：优先由头部接近球路、躯干前压和头部运动确认，不只依赖球速"
    if phase == "大幅摆腿/抽射候选":
        return "技巧原因：摆腿幅度、腿部伸展和支撑形态已形成击球窗口，球速作为后验确认"
    if phase == "触球/击球瞬间":
        return f"释放判断：球速 {ball_speed:.0f}px/s，球距变化{signed_value(ball_release)}px，说明力量已从脚传到球"
    if phase == "持球推进/调整":
        return f"技巧原因：球距 {ball_to_foot:.0f}px 且触球 {contact:.0f}/100，更像带球调整而非最终击球"
    if phase == "射门准备":
        return "技巧原因：身体和摆腿已准备，但球还未出现高速释放，先不判为射门关键帧"
    if phase == "摆腿加速":
        return "技巧原因：摆腿速度上升，等待球距拉开或球速突增来确认击球"
    return "技巧原因：当前画面缺少稳定的近球与释放证据"


def action_evidence_label(
    shot_type: str,
    leg_score: float,
    header_score: float,
    ball_to_foot: float,
    head_to_ball: float,
    ball_speed: float,
) -> str:
    if shot_type == "头球候选":
        distance = "未知" if head_to_ball <= 0.0 else f"{head_to_ball:.0f}px"
        return f"动作证据：头球 {header_score:.0f}/100，头部距球 {distance}，结合上体前压判断攻门窗口"
    if shot_type in {"大幅摆腿/抽射候选", "脚下射门候选"}:
        distance = "未知" if ball_to_foot <= 0.0 else f"{ball_to_foot:.0f}px"
        return f"动作证据：脚下射门 {leg_score:.0f}/100，脚到球 {distance}，球速 {ball_speed:.0f}px/s 仅作确认"
    return f"动作证据：脚下 {leg_score:.0f}/100，头球 {header_score:.0f}/100，尚未形成稳定射门窗口"


def sliding_tackle_proxy(pose: PersonPose, torso_lean: float) -> bool:
    x1, y1, x2, y2 = pose.bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    left_knee = joint_angle(pose, "left_hip", "left_knee", "left_ankle")
    right_knee = joint_angle(pose, "right_hip", "right_knee", "right_ankle")
    extended_leg = max(left_knee, right_knee) > 150
    return bool((w / h > 1.15 or torso_lean > 58) and extended_leg)


def bbox_width_height_ratio(bbox: tuple[float, float, float, float]) -> float:
    return max(1.0, bbox[2] - bbox[0]) / max(1.0, bbox[3] - bbox[1])


def is_goalkeeper_save_like_pose(pose: PersonPose, ball: BallBox | None, torso_lean: float) -> bool:
    body_ratio = bbox_width_height_ratio(pose.bbox)
    extension = goalkeeper_extension_score(pose)
    dive = goalkeeper_dive_score(torso_lean, body_ratio, 0.0, extension)
    ball_dist = nearest_hand_or_body_distance(pose, ball)
    ball_near_save_area = ball_dist > 0.0 and ball_dist <= max(155.0, bbox_diag(pose.bbox) * 0.62)
    return bool(
        ball_near_save_area
        and (
            dive >= 56.0
            or body_ratio >= 1.12
            or (torso_lean >= 45.0 and extension >= 52.0)
        )
    )


def goalkeeper_extension_score(pose: PersonPose) -> float:
    ls = keypoint(pose, "left_shoulder")
    rs = keypoint(pose, "right_shoulder")
    lw = keypoint(pose, "left_wrist")
    rw = keypoint(pose, "right_wrist")
    diag = bbox_diag(pose.bbox)
    values = []
    if ls is not None and lw is not None:
        values.append(pixel_dist(ls, lw) / max(1.0, diag))
    if rs is not None and rw is not None:
        values.append(pixel_dist(rs, rw) / max(1.0, diag))
    if not values:
        return 0.0
    return float(max(0.0, min(100.0, max(values) * 185.0)))


def goalkeeper_set_position_score(pose: PersonPose, torso_lean: float, body_ratio: float) -> float:
    knees = [joint_angle(pose, "left_hip", "left_knee", "left_ankle"), joint_angle(pose, "right_hip", "right_knee", "right_ankle")]
    knee_ready = 0.0
    valid_knees = [value for value in knees if value > 0.0]
    if valid_knees:
        avg_knee = float(np.mean(valid_knees))
        knee_ready = max(0.0, min(1.0, 1.0 - abs(avg_knee - 135.0) / 55.0))
    upright = max(0.0, min(1.0, 1.0 - max(0.0, body_ratio - 0.95) / 0.65))
    lean_ready = max(0.0, min(1.0, 1.0 - abs(torso_lean - 12.0) / 42.0))
    return float(100.0 * (0.42 * knee_ready + 0.34 * upright + 0.24 * lean_ready))


def goalkeeper_dive_score(torso_lean: float, body_ratio: float, center_speed: float, extension_score: float) -> float:
    horizontal = max(0.0, min(1.0, (body_ratio - 0.82) / 0.95))
    lean = max(0.0, min(1.0, (torso_lean - 28.0) / 45.0))
    motion = max(0.0, min(1.0, center_speed / 95.0))
    extension = max(0.0, min(1.0, extension_score / 100.0))
    return float(100.0 * (0.36 * horizontal + 0.28 * lean + 0.20 * motion + 0.16 * extension))


def nearest_hand_or_body_distance(pose: PersonPose, ball: BallBox | None) -> float:
    if ball is None:
        return 0.0
    points = [
        keypoint(pose, "left_wrist"),
        keypoint(pose, "right_wrist"),
        keypoint(pose, "left_ankle"),
        keypoint(pose, "right_ankle"),
        bbox_center(pose.bbox),
    ]
    dists = [pixel_dist(point, ball.center) for point in points if point is not None]
    return min(dists) if dists else 0.0


def goalkeeper_phase(dive_score: float, set_position_score: float, reaction_sec: float) -> str:
    if dive_score >= 74.0:
        return "倒地/侧扑扑救"
    if dive_score >= 55.0:
        return "扑救启动/横向移动"
    if reaction_sec >= 0.0 and set_position_score >= 55.0:
        return "射门后反应"
    if set_position_score >= 55.0:
        return "准备站位"
    return "观察/调整站位"


def goalkeeper_technique_line(phase: str, dive: float, set_pos: float, extension: float) -> str:
    if phase == "倒地/侧扑扑救":
        return f"规范性：重心倒向球路，手臂伸展 {extension:.0f}/100，侧扑覆盖面较明确"
    if phase == "扑救启动/横向移动":
        return f"规范性：横向启动已出现，扑救强度 {dive:.0f}/100，需要继续观察落地与封堵"
    if phase == "准备站位":
        return f"规范性：准备姿态 {set_pos:.0f}/100，膝髋保持弹性，便于二次反应"
    return "规范性：当前更像站位调整，尚未进入明确扑救动作"


def power_label(ball_speed: float) -> str:
    if ball_speed <= 0:
        return "未检测到球速"
    if ball_speed < 450:
        return "低速/控制型"
    if ball_speed < 1100:
        return "中等力量"
    return "高速射门候选"


def foot_candidate(left_activity: float, right_activity: float) -> str:
    total = left_activity + right_activity
    if total <= 1e-6:
        return "暂不明确"
    diff_ratio = abs(left_activity - right_activity) / total
    if diff_ratio < 0.035:
        return "暂不明确"
    side = "左脚" if left_activity > right_activity else "右脚"
    confidence = "较强" if diff_ratio >= 0.12 else "弱"
    return f"{side}候选({confidence})"


def smooth_pose(previous: PersonPose | None, current: PersonPose | None, alpha: float) -> PersonPose | None:
    if current is None:
        return None
    if previous is None:
        return current
    alpha = max(0.05, min(1.0, alpha))
    bbox = tuple(float(alpha * c + (1.0 - alpha) * p) for p, c in zip(previous.bbox, current.bbox))
    keypoints = alpha * current.keypoints + (1.0 - alpha) * previous.keypoints
    conf = np.maximum(current.conf, previous.conf * 0.92)
    score = float(alpha * current.score + (1.0 - alpha) * previous.score)
    return PersonPose(bbox, score, keypoints, conf)


def select_primary_pose(
    poses: list[PersonPose],
    previous: PersonPose | None,
    ball: BallBox | None = None,
    frame: np.ndarray | None = None,
    subject_uniform: str = "auto",
    excluded: list[PersonPose] | None = None,
) -> PersonPose | None:
    candidates = exclude_poses(poses, excluded)
    if not candidates:
        return None
    if previous is not None:
        locked = best_continuity_pose(candidates, previous)
        if locked is not None:
            if ball is None:
                if subject_uniform != "auto":
                    best_uniform_pose = max(candidates, key=lambda p: pose_frame_rank(p, frame, subject_uniform))
                    locked_uniform = uniform_multiplier(frame, locked.bbox, subject_uniform)
                    best_uniform = uniform_multiplier(frame, best_uniform_pose.bbox, subject_uniform)
                    if best_uniform_pose is not locked and locked_uniform < 0.90 and best_uniform >= max(1.05, locked_uniform * 1.75):
                        return best_uniform_pose
                return locked
            best_ball_pose = max(
                candidates,
                key=lambda p: pose_ball_rank(p, ball, None, frame, subject_uniform),
            )
            locked_dist = pose_ball_distance(locked, ball)
            best_dist = pose_ball_distance(best_ball_pose, ball)
            locked_uniform = uniform_multiplier(frame, locked.bbox, subject_uniform)
            best_uniform = uniform_multiplier(frame, best_ball_pose.bbox, subject_uniform)
            best_uniform_pose = max(candidates, key=lambda p: pose_frame_rank(p, frame, subject_uniform))
            best_frame_uniform = uniform_multiplier(frame, best_uniform_pose.bbox, subject_uniform)
            switch_dist_limit = 260.0 if locked_uniform < 0.90 else 95.0
            should_switch = (
                best_ball_pose is not locked
                and best_dist > 0.0
                and best_dist <= switch_dist_limit
                and (locked_dist <= 0.0 or best_dist + max(70.0, locked_dist * 0.25) < locked_dist)
                and not (0.0 < locked_dist <= max(85.0, bbox_diag(locked.bbox) * 0.42))
                and best_uniform >= locked_uniform * 0.80
            )
            uniform_correction = (
                subject_uniform != "auto"
                and best_uniform_pose is not locked
                and locked_uniform < 0.90
                and best_frame_uniform >= max(1.05, locked_uniform * 1.75)
                and (ball is None or pose_ball_distance(best_uniform_pose, ball) <= locked_dist + 160.0)
            )
            if not should_switch:
                return best_uniform_pose if uniform_correction else locked
            return best_ball_pose
    if ball is not None:
        return max(
            candidates,
            key=lambda p: pose_ball_rank(p, ball, previous, frame, subject_uniform),
        )
    if previous is None:
        return max(candidates, key=lambda p: pose_frame_rank(p, frame, subject_uniform))
    prev_center = bbox_center(previous.bbox)
    return max(
        candidates,
        key=lambda p: (
            (p.bbox[2] - p.bbox[0])
            * (p.bbox[3] - p.bbox[1])
            * p.score
            * uniform_multiplier(frame, p.bbox, subject_uniform)
            / (1.0 + 0.01 * pixel_dist(bbox_center(p.bbox), prev_center))
        ),
    )


def exclude_poses(poses: list[PersonPose], excluded: list[PersonPose] | None, iou_threshold: float = 0.08) -> list[PersonPose]:
    if not excluded:
        return poses
    output = []
    for pose in poses:
        if any(item is not None and bbox_iou(pose.bbox, item.bbox) > iou_threshold for item in excluded):
            continue
        output.append(pose)
    return output or poses


def select_goalkeeper_pose(
    poses: list[PersonPose],
    previous: PersonPose | None,
    shooter: PersonPose | None,
    frame: np.ndarray | None,
    goalkeeper_uniform: str,
    ball: BallBox | None = None,
) -> PersonPose | None:
    candidates = []
    for pose in poses:
        if shooter is not None and bbox_iou(pose.bbox, shooter.bbox) > 0.08:
            continue
        candidates.append(pose)
    if not candidates:
        return None
    if previous is not None:
        locked = best_continuity_pose(candidates, previous)
        if locked is not None and uniform_multiplier(frame, locked.bbox, goalkeeper_uniform) >= 0.55:
            return locked
    return max(candidates, key=lambda p: goalkeeper_rank(p, frame, goalkeeper_uniform, ball))


def goalkeeper_rank(pose: PersonPose, frame: np.ndarray | None, goalkeeper_uniform: str, ball: BallBox | None = None) -> float:
    area = max(1.0, (pose.bbox[2] - pose.bbox[0]) * (pose.bbox[3] - pose.bbox[1]))
    torso_lean = torso_lean_deg(pose)
    body_ratio = max(1.0, pose.bbox[2] - pose.bbox[0]) / max(1.0, pose.bbox[3] - pose.bbox[1])
    dive = goalkeeper_dive_score(torso_lean, body_ratio, 0.0, goalkeeper_extension_score(pose))
    hand_dist = nearest_hand_or_body_distance(pose, ball) if ball is not None else 0.0
    hand_bonus = 1.0 + (1.35 / (1.0 + 0.010 * hand_dist) if hand_dist > 0.0 else 0.0)
    action_bonus = 1.0 + 0.010 * dive
    uniform = uniform_multiplier(frame, pose.bbox, goalkeeper_uniform)
    if goalkeeper_uniform == "auto" and dive < 52.0 and hand_dist > 180.0:
        uniform *= 0.55
    return float((area ** 0.45) * pose.score * action_bonus * hand_bonus * uniform)


def pose_frame_rank(pose: PersonPose, frame: np.ndarray | None, subject_uniform: str) -> float:
    area = max(1.0, (pose.bbox[2] - pose.bbox[0]) * (pose.bbox[3] - pose.bbox[1]))
    lower_visible = sum(1 for name in ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"] if keypoint(pose, name, 0.18) is not None)
    lower_bonus = 1.0 + 0.10 * lower_visible
    return float((area ** 0.52) * pose.score * lower_bonus * uniform_multiplier(frame, pose.bbox, subject_uniform))


def best_continuity_pose(poses: list[PersonPose], previous: PersonPose) -> PersonPose | None:
    prev_center = bbox_center(previous.bbox)
    prev_diag = bbox_diag(previous.bbox)

    def continuity_score(pose: PersonPose) -> float:
        center_dist = pixel_dist(bbox_center(pose.bbox), prev_center)
        iou = bbox_iou(pose.bbox, previous.bbox)
        area = max(1.0, (pose.bbox[2] - pose.bbox[0]) * (pose.bbox[3] - pose.bbox[1]))
        return float((1.0 + 7.0 * iou) * pose.score * (area ** 0.18) / (1.0 + center_dist / max(80.0, prev_diag * 0.35)))

    candidate = max(poses, key=continuity_score)
    center_dist = pixel_dist(bbox_center(candidate.bbox), prev_center)
    iou = bbox_iou(candidate.bbox, previous.bbox)
    if iou >= 0.035 or center_dist <= max(220.0, prev_diag * 0.55):
        return candidate
    return None


def pose_ball_distance(pose: PersonPose, ball: BallBox) -> float:
    foot_dist = nearest_foot_distance(pose, ball)
    if foot_dist > 0.0:
        return foot_dist
    return pixel_dist(bbox_center(pose.bbox), ball.center)


def pose_ball_rank(pose: PersonPose, ball: BallBox, previous: PersonPose | None, frame: np.ndarray | None, subject_uniform: str) -> float:
    area = max(1.0, (pose.bbox[2] - pose.bbox[0]) * (pose.bbox[3] - pose.bbox[1]))
    foot_dist = pose_ball_distance(pose, ball)
    proximity = 1.0 / (1.0 + 0.012 * foot_dist)
    lower_visible = sum(1 for name in ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"] if keypoint(pose, name, 0.18) is not None)
    lower_bonus = 1.0 + 0.10 * lower_visible
    if previous is not None:
        center_dist = pixel_dist(bbox_center(pose.bbox), bbox_center(previous.bbox))
        continuity = (1.0 + 5.0 * bbox_iou(pose.bbox, previous.bbox)) / (1.0 + 0.004 * center_dist)
    else:
        continuity = 1.0
    return float((area ** 0.45) * pose.score * proximity * lower_bonus * continuity * uniform_multiplier(frame, pose.bbox, subject_uniform))


def uniform_multiplier(frame: np.ndarray | None, bbox: tuple[float, float, float, float], preference: str) -> float:
    if frame is None or preference == "auto":
        return 1.0
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    bw = x2 - x1
    bh = y2 - y1
    tx1 = x1 + int(bw * 0.22)
    tx2 = x1 + int(bw * 0.78)
    ty1 = y1 + int(bh * 0.16)
    ty2 = y1 + int(bh * 0.62)
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return 1.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    if preference == "dark":
        dark_ratio = float((val < 112).mean())
        red_ratio = float((((hue < 12) | (hue > 165)) & (sat > 70) & (val > 75)).mean())
        white_ratio = float(((sat < 45) & (val > 135)).mean())
        score = dark_ratio - 1.05 * red_ratio - 0.70 * white_ratio
    elif preference == "light":
        score = float(((sat < 65) & (val > 135)).mean())
    elif preference == "red":
        score = float((((hue < 12) | (hue > 165)) & (sat > 70) & (val > 70)).mean())
    elif preference == "blue":
        score = float(((hue > 90) & (hue < 132) & (sat > 55)).mean())
    elif preference == "yellow":
        score = float(((hue > 18) & (hue < 42) & (sat > 55)).mean())
    elif preference == "yellowgreen":
        score = float(((hue > 18) & (hue < 86) & (sat > 45) & (val > 80)).mean())
    else:
        score = 0.0
    return 0.25 + 4.75 * max(0.0, min(1.0, score))


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2))


def bbox_diag(bbox: tuple[float, float, float, float]) -> float:
    return float(math.hypot(max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])))


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(1.0, area_a + area_b - inter))


def detect_poses(model: YOLO, frame: np.ndarray, imgsz: int, conf: float) -> list[PersonPose]:
    result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    if result.boxes is None or result.keypoints is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    xy = result.keypoints.xy.cpu().numpy()
    kconf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.ones(xy.shape[:2], dtype=np.float32)
    return [PersonPose(tuple(map(float, box)), float(score), keypoints, confs) for box, score, keypoints, confs in zip(boxes, scores, xy, kconf)]


def detect_ball(model: YOLO, frame: np.ndarray, imgsz: int, conf: float) -> BallBox | None:
    result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    if result.boxes is None:
        return None
    candidates = []
    frame_area = float(frame.shape[0] * frame.shape[1])
    for box, score, cls_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        if int(cls_id) != 32:
            continue
        x1, y1, x2, y2 = map(float, box)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        area = w * h
        aspect = w / h
        if area > frame_area * 0.006 or area < 12.0:
            continue
        if not 0.45 <= aspect <= 2.2:
            continue
        candidates.append(BallBox(((x1 + x2) / 2, (y1 + y2) / 2), (x1, y1, x2, y2), float(score)))
    if not candidates:
        return None
    return max(candidates, key=lambda b: b.score)


def draw_pose(frame: np.ndarray, pose: PersonPose | None, metrics: dict[str, float | str | bool], draw_label: bool = False) -> None:
    if pose is None:
        return
    x1, y1, x2, y2 = [int(v) for v in pose.bbox]
    color = (70, 220, 80) if bool(metrics.get("shot_candidate")) else (80, 180, 255)
    if bool(metrics.get("tackle_risk")):
        color = COLORS["danger"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    if draw_label:
        put_text(frame, "主体球员", (x1, max(24, y1 - 8)), scale=0.48, color=color)
    for a_name, b_name in SKELETON:
        a = keypoint(pose, a_name)
        b = keypoint(pose, b_name)
        if a is None or b is None:
            continue
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 3, cv2.LINE_AA)
    for name in COCO_KPTS:
        point = keypoint(pose, name)
        if point is not None:
            cv2.circle(frame, (int(point[0]), int(point[1])), 4, (250, 250, 250), -1, cv2.LINE_AA)


def draw_goalkeeper_pose(frame: np.ndarray, pose: PersonPose | None, metrics: dict[str, float | str | bool], draw_label: bool = False) -> None:
    if pose is None or not bool(metrics.get("goalkeeper_visible")):
        return
    x1, y1, x2, y2 = [int(v) for v in pose.bbox]
    color = (80, 255, 180) if not bool(metrics.get("save_attempt")) else (40, 245, 255)
    if bool(metrics.get("save_success_proxy")):
        color = (70, 220, 80)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    if draw_label:
        put_text(frame, "门将", (x1, max(24, y1 - 8)), scale=0.48, color=color)
    for a_name, b_name in SKELETON:
        a = keypoint(pose, a_name)
        b = keypoint(pose, b_name)
        if a is None or b is None:
            continue
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 3, cv2.LINE_AA)
    for name in COCO_KPTS:
        point = keypoint(pose, name)
        if point is not None:
            cv2.circle(frame, (int(point[0]), int(point[1])), 4, (245, 255, 245), -1, cv2.LINE_AA)


def draw_ball(frame: np.ndarray, ball: BallBox | None, draw_label: bool = False) -> None:
    if ball is None:
        return
    x1, y1, x2, y2 = [int(v) for v in ball.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["ball"], 2)
    cv2.circle(frame, (int(ball.center[0]), int(ball.center[1])), 5, COLORS["ball"], -1, cv2.LINE_AA)
    if draw_label:
        put_text(frame, "球", (x1, max(24, y1 - 8)), scale=0.48, color=COLORS["ball"])


def draw_panel(frame: np.ndarray, frame_idx: int, fps: float, state: IndividualTechniqueState, metrics: dict[str, float | str | bool]) -> None:
    x1, y1, x2, y2 = 24, frame.shape[0] - 318, 835, frame.shape[0] - 24
    blend_rect(frame, x1, y1, x2, y2, COLORS["panel"], 0.74)
    verdict = state.latest_verdict if is_key_technique_frame(metrics) else state.display_verdict
    notes = state.latest_notes if is_key_technique_frame(metrics) else state.display_notes
    put_text(frame, "个人进攻技术分析", (x1 + 16, y1 + 34), scale=0.72, thickness=2)
    put_text(frame, verdict, (x1 + 16, y1 + 72), scale=0.58, thickness=1)
    put_text(frame, f"时间 {frame_idx / fps:05.2f}s  姿态采样 {state.pose_visible_frames}/{state.frame_count}  足球采样 {state.ball_visible_frames}/{state.frame_count}", (x1 + 16, y1 + 104), scale=0.44)
    for idx, note in enumerate(notes[:4]):
        put_text(frame, note, (x1 + 16, y1 + 136 + idx * 28), scale=0.44)
    put_text(frame, "专业分项", (x1 + 16, y1 + 254), scale=0.46, thickness=1)
    draw_score_bar(frame, "平衡", float(metrics.get("balance_score", 0.0)), (x1 + 100, y1 + 240), 110)
    draw_score_bar(frame, "支撑", float(metrics.get("plant_score", 0.0)), (x1 + 255, y1 + 240), 110)
    draw_score_bar(frame, "摆腿", float(metrics.get("swing_score", 0.0)), (x1 + 410, y1 + 240), 110)
    draw_score_bar(frame, "触球", float(metrics.get("contact_score", 0.0)), (x1 + 565, y1 + 240), 110)
    phase = str(metrics.get("phase", "未知阶段"))
    put_text(frame, f"阶段统计：射门候选 {state.shot_candidate_frames}  铲抢风险 {state.tackle_risk_frames}  当前 {phase}  最大球速代理 {state.max_ball_speed:.0f}px/s", (x1 + 16, y1 + 292), scale=0.40)


def draw_goalkeeper_panel(frame: np.ndarray, frame_idx: int, fps: float, state: GoalkeeperTechniqueState, metrics: dict[str, float | str | bool]) -> None:
    x2 = frame.shape[1] - 24
    x1 = max(860, x2 - 760)
    y1, y2 = frame.shape[0] - 274, frame.shape[0] - 24
    blend_rect(frame, x1, y1, x2, y2, COLORS["panel"], 0.70)
    active_goalkeeper_action = (
        bool(metrics.get("save_attempt"))
        or float(metrics.get("dive_score", 0.0)) >= 55.0
        or float(metrics.get("reaction_sec", -1.0)) >= 0.0
    )
    verdict = state.latest_verdict if active_goalkeeper_action else state.display_verdict
    notes = state.latest_notes if active_goalkeeper_action else state.display_notes
    put_text(frame, "门将防守技术分析", (x1 + 16, y1 + 32), scale=0.64, thickness=2, color=(110, 245, 220))
    put_text(frame, verdict, (x1 + 16, y1 + 68), scale=0.52, thickness=1)
    put_text(frame, f"时间 {frame_idx / fps:05.2f}s  门将采样 {state.visible_frames}/{state.frame_count}  扑救尝试 {state.save_attempts}", (x1 + 16, y1 + 98), scale=0.40)
    for idx, note in enumerate(notes[:4]):
        put_text(frame, note, (x1 + 16, y1 + 128 + idx * 25), scale=0.39)
    draw_score_bar(frame, "准备", float(metrics.get("set_position_score", 0.0)), (x1 + 16, y2 - 36), 95)
    draw_score_bar(frame, "扑救", float(metrics.get("dive_score", 0.0)), (x1 + 170, y2 - 36), 95)
    draw_score_bar(frame, "伸展", float(metrics.get("extension_score", 0.0)), (x1 + 324, y2 - 36), 95)
    save_rate = state.successful_save_proxies / state.save_attempts if state.save_attempts else 0.0
    put_text(frame, f"扑救成功代理 {state.successful_save_proxies}/{state.save_attempts} ({save_rate * 100:.0f}%)", (x1 + 486, y2 - 18), scale=0.36)


def draw_score_bar(frame: np.ndarray, label: str, value: float, origin: tuple[int, int], width: int) -> None:
    x, y = origin
    value = max(0.0, min(100.0, value))
    color = (70, 220, 80) if value >= 70 else (45, 200, 245) if value >= 45 else COLORS["danger"]
    put_text(frame, label, (x, y + 18), scale=0.36)
    cv2.rectangle(frame, (x + 42, y + 5), (x + 42 + width, y + 17), (80, 88, 96), 1)
    cv2.rectangle(frame, (x + 43, y + 6), (x + 43 + int(width * value / 100.0), y + 16), color, -1)
    put_text(frame, f"{value:.0f}", (x + 50 + width, y + 18), scale=0.34, color=color)


def is_key_technique_frame(metrics: dict[str, float | str | bool]) -> bool:
    if bool(metrics.get("goalkeeper_like")):
        return False
    phase = str(metrics.get("phase", ""))
    leg_score = metric_float(metrics, "leg_shot_score")
    header_score = metric_float(metrics, "header_score")
    shot_action = metric_float(metrics, "shot_action_score")
    ball_speed = metric_float(metrics, "ball_speed")
    ball_to_foot = metric_float(metrics, "ball_to_foot", 999.0)
    head_to_ball = metric_float(metrics, "head_to_ball", 999.0)
    if header_score >= 72.0 and head_to_ball <= 150.0:
        return True
    if leg_score >= 76.0 and ball_to_foot <= 210.0:
        return True
    if ball_speed >= 2600.0 and metric_float(metrics, "posture_score") >= 54.0 and metric_float(metrics, "swing_score") >= 48.0:
        return True
    if shot_action >= 78.0 and (ball_speed >= 520.0 or bool(metrics.get("near_ball"))):
        return True
    if phase == "触球/击球瞬间" and bool(metrics.get("release_detected")):
        return True
    if (
        phase == "摆腿加速"
        and bool(metrics.get("shot_candidate"))
        and metric_float(metrics, "posture_score") >= 76
        and ball_speed >= 900.0
        and ball_to_foot <= 90.0
    ):
        return True
    return False


def is_strong_technique_frame(metrics: dict[str, float | str | bool]) -> bool:
    return bool(
        (
            bool(metrics.get("release_detected"))
            and metric_float(metrics, "ball_speed") >= 1200.0
            and metric_float(metrics, "posture_score") >= 75.0
        )
        or metric_float(metrics, "leg_shot_score") >= 84.0
        or metric_float(metrics, "header_score") >= 78.0
    )


def keyframe_tier(metrics: dict[str, float | str | bool]) -> int:
    if not is_key_technique_frame(metrics):
        return 0
    if is_high_value_shot_frame(metrics):
        return 2
    return 1


def pre_contact_keyframe_score(metrics: dict[str, float | str | bool], trigger_frame: int, candidate_frame: int) -> float:
    ball_to_foot = float(metrics.get("ball_to_foot", 0.0))
    head_to_ball = metric_float(metrics, "head_to_ball")
    if ball_to_foot <= 0.0 and head_to_ball <= 0.0:
        return -1e6
    posture = metric_float(metrics, "posture_score")
    plant = metric_float(metrics, "plant_score")
    swing = metric_float(metrics, "swing_score")
    contact = metric_float(metrics, "contact_score")
    ball_speed = metric_float(metrics, "ball_speed")
    leg_score = metric_float(metrics, "leg_shot_score")
    header_score = metric_float(metrics, "header_score")
    action_score = metric_float(metrics, "shot_action_score")
    phase = str(metrics.get("phase", ""))
    frame_gap = max(0, trigger_frame - candidate_frame)

    close_to_foot = 0.0 if ball_to_foot <= 0.0 else max(0.0, 100.0 - abs(ball_to_foot - 42.0) * 1.35)
    close_to_head = 0.0 if head_to_ball <= 0.0 else max(0.0, 100.0 - abs(head_to_ball - 52.0) * 1.05)
    not_too_late = max(0.0, 100.0 - ball_speed / 32.0)
    score = (
        1.65 * contact
        + 1.25 * close_to_foot
        + 1.15 * swing
        + 0.75 * posture
        + 0.45 * plant
        + 1.15 * close_to_head
        + 1.05 * leg_score
        + 1.15 * header_score
        + 0.70 * action_score
        + 0.25 * not_too_late
        - 1.6 * frame_gap
    )
    if phase in {"摆腿加速", "触球/击球瞬间", "大幅摆腿/抽射候选", "头球攻门候选"}:
        score += 38.0
    elif phase == "射门准备":
        score += 18.0
    if bool(metrics.get("near_ball")):
        score += 25.0
    if bool(metrics.get("release_detected")) and ball_to_foot >= 150.0:
        score -= 90.0
    return float(score)


def select_explanation_segment(
    history: deque[KeyframeSnapshot],
    selected_frame: int,
    trigger_frame: int,
    max_segment_frames: int,
) -> list[KeyframeSnapshot]:
    if max_segment_frames <= 1:
        return []
    start_frame = max(selected_frame - max(1, max_segment_frames // 3), trigger_frame - max_segment_frames)
    end_frame = min(trigger_frame, selected_frame + max_segment_frames)
    segment = [item for item in history if start_frame <= item.frame_idx <= end_frame]
    if len(segment) <= 1:
        return []
    return segment[:max_segment_frames]


def select_explanation_snapshot(
    history: deque[KeyframeSnapshot],
    trigger_frame: int,
    trigger_tier: int,
    max_lookback_frames: int,
) -> KeyframeSnapshot | None:
    if not history:
        return None
    eligible = [
        item
        for item in history
        if item.frame_idx <= trigger_frame and trigger_frame - item.frame_idx <= max(1, max_lookback_frames)
    ]
    if not eligible:
        return history[-1]
    if trigger_tier >= 2:
        contact_candidates = [item for item in eligible if float(item.metrics.get("contact_score", 0.0)) >= 35.0]
        if contact_candidates:
            eligible = contact_candidates
    return max(eligible, key=lambda item: pre_contact_keyframe_score(item.metrics, trigger_frame, item.frame_idx))


def is_high_value_shot_frame(metrics: dict[str, float | str | bool]) -> bool:
    if bool(metrics.get("goalkeeper_like")):
        return False
    ball_speed = metric_float(metrics, "ball_speed")
    ball_to_foot = metric_float(metrics, "ball_to_foot")
    posture = metric_float(metrics, "posture_score")
    plant = metric_float(metrics, "plant_score")
    swing = metric_float(metrics, "swing_score")
    leg_score = metric_float(metrics, "leg_shot_score")
    header_score = metric_float(metrics, "header_score")
    head_to_ball = metric_float(metrics, "head_to_ball")
    if header_score >= 82.0 and 0.0 < head_to_ball <= 125.0:
        return True
    if leg_score >= 82.0 and 0.0 < ball_to_foot <= 210.0 and swing >= 52.0:
        return True
    if ball_speed >= 3000.0 and posture >= 54.0 and swing >= 48.0 and plant >= 48.0:
        return True
    return bool(
        posture >= 74.0
        and plant >= 55.0
        and (ball_speed >= 2800.0 or leg_score >= 82.0)
        and (ball_to_foot >= 180.0 or swing >= 52.0)
    )


def keyframe_explain_lines(metrics: dict[str, float | str | bool], state: IndividualTechniqueState) -> list[str]:
    left_speed = float(metrics.get("left_ankle_speed", 0.0))
    right_speed = float(metrics.get("right_ankle_speed", 0.0))
    kick_side = "左脚" if left_speed > right_speed else "右脚"
    lines = state.professional_analysis_lines(metrics, compact=False)
    tier = int(metrics.get("trigger_tier", keyframe_tier(metrics)))
    shot_type = str(metrics.get("shot_type", "未判定"))
    tier_text = "高价值射门回看帧" if tier >= 2 else "普通触球候选帧"
    output = [
        f"{tier_text}：{metrics.get('phase', '射门动作')}，类型 {shot_type}，{kick_side}发力更明显",
        *lines[:5],
    ]
    review_gap = int(metrics.get("keyframe_review_gap", 0))
    if review_gap > 0:
        output.append(f"片段机制：由球速/摆腿/头球形态联合确认，当前画面提前 {review_gap} 帧观察动作窗口")
    return output


def draw_keyframe_explanation(frame: np.ndarray, metrics: dict[str, float | str | bool], state: IndividualTechniqueState) -> None:
    x1, y1, x2, y2 = 440, 64, frame.shape[1] - 180, 358
    blend_rect(frame, x1, y1, x2, y2, COLORS["panel"], 0.80)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 235, 255), 2)
    put_text(frame, "射门关键动作片段", (x1 + 22, y1 + 42), scale=0.76, thickness=2, color=(90, 235, 255))
    cv2.rectangle(frame, (x2 - 132, y1 + 14), (x2 - 20, y1 + 48), COLORS["danger"], -1)
    put_text(frame, "暂停讲解", (x2 - 122, y1 + 40), scale=0.46, thickness=2)
    for idx, line in enumerate(keyframe_explain_lines(metrics, state)):
        put_text(frame, line, (x1 + 22, y1 + 84 + idx * 29), scale=0.43)
    put_text(frame, "此处自动延缓，便于观察支撑脚、摆腿轨迹和触球时机", (x1 + 22, y2 - 22), scale=0.40, color=(220, 225, 230))


def main() -> int:
    args = parse_args()
    input_video = (PROJECT_ROOT / args.input_video).resolve()
    output_video = (PROJECT_ROOT / args.output_video).resolve()
    report_path = output_video.with_suffix(".individual_report.json")
    snapshot_path = output_video.with_name(f"{output_video.stem}_frame{args.snapshot_frame:03d}.jpg")

    pose_model = YOLO(args.pose_model)
    ball_model = YOLO(args.ball_model)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(args.start_sec * fps))
    max_frames = min(max(1, int(args.duration_sec * fps)), max(0, total_frames - start_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")

    state = IndividualTechniqueState(fps)
    goalkeeper_state = GoalkeeperTechniqueState(fps)
    current_pose: PersonPose | None = None
    render_pose: PersonPose | None = None
    current_goalkeeper_pose: PersonPose | None = None
    render_goalkeeper_pose: PersonPose | None = None
    current_ball: BallBox | None = None
    current_metrics: dict[str, float | str | bool] = {}
    current_goalkeeper_metrics: dict[str, float | str | bool] = {}
    wrote_snapshot = False
    last_keyframe_pause = -10**9
    pause_frames = max(0, int(args.keyframe_pause_sec * fps))
    minor_pause_frames = max(0, int(args.minor_keyframe_pause_sec * fps))
    keyframe_cooldown = max(1, int(args.keyframe_cooldown_sec * fps))
    keyframe_lookback_frames = max(1, int(args.keyframe_lookback_sec * fps))
    keyframe_segment_frames = max(0, int(args.keyframe_segment_sec * fps))
    keyframe_history: deque[KeyframeSnapshot] = deque(maxlen=keyframe_lookback_frames + keyframe_segment_frames + 6)
    written_frames = 0
    inserted_keyframes = 0
    skipped_minor_keyframes = 0

    for local_idx in range(1, max_frames + 1):
        ok, frame = cap.read()
        if not ok:
            break
        source_frame = start_frame + local_idx
        if local_idx == 1 or local_idx % max(1, args.infer_step) == 0:
            poses = detect_poses(pose_model, frame, args.imgsz, args.pose_conf)
            current_ball = detect_ball(ball_model, frame, args.imgsz, args.ball_conf)
            if not args.disable_goalkeeper_analysis:
                current_goalkeeper_pose = select_goalkeeper_pose(poses, current_goalkeeper_pose, None, frame, args.goalkeeper_uniform, current_ball)
            current_pose = select_primary_pose(
                poses,
                current_pose,
                current_ball,
                frame,
                args.subject_uniform,
                [current_goalkeeper_pose] if not args.disable_goalkeeper_analysis and current_goalkeeper_pose is not None else None,
            )
            current_metrics = state.update(source_frame, current_pose, current_ball)
            if not args.disable_goalkeeper_analysis:
                current_goalkeeper_pose = select_goalkeeper_pose(poses, current_goalkeeper_pose, current_pose, frame, args.goalkeeper_uniform, current_ball)
                current_goalkeeper_metrics = goalkeeper_state.update(source_frame, current_goalkeeper_pose, current_ball, current_metrics)
                render_goalkeeper_pose = smooth_pose(render_goalkeeper_pose, current_goalkeeper_pose, args.smooth_alpha)
            render_pose = smooth_pose(render_pose, current_pose, args.smooth_alpha)

        out = frame.copy()
        draw_pose(out, render_pose, current_metrics, args.draw_moving_labels)
        if not args.disable_goalkeeper_analysis:
            draw_goalkeeper_pose(out, render_goalkeeper_pose, current_goalkeeper_metrics, args.draw_moving_labels)
        draw_ball(out, current_ball, args.draw_moving_labels)
        draw_panel(out, source_frame, fps, state, current_metrics)
        if not args.disable_goalkeeper_analysis:
            draw_goalkeeper_panel(out, source_frame, fps, goalkeeper_state, current_goalkeeper_metrics)
        writer.write(out)
        written_frames += 1
        if local_idx == args.snapshot_frame:
            cv2.imwrite(str(snapshot_path), out)
            wrote_snapshot = True
        tier = keyframe_tier(current_metrics)
        keyframe_history.append(KeyframeSnapshot(source_frame, out.copy(), current_metrics.copy(), tier))
        tier_pause_frames = pause_frames
        if args.keyframe_pause_policy == "high-value" and tier == 1:
            tier_pause_frames = minor_pause_frames
        if (
            tier_pause_frames > 0
            and inserted_keyframes < args.max_keyframes
            and tier > 0
            and (
                source_frame - last_keyframe_pause >= keyframe_cooldown
                or (
                    tier >= 2
                    and source_frame - last_keyframe_pause >= max(1, int(0.65 * fps))
                )
            )
        ):
            selected = select_explanation_snapshot(keyframe_history, source_frame, tier, keyframe_lookback_frames)
            freeze = (selected.image.copy() if selected is not None else out.copy())
            explanation_metrics = (selected.metrics.copy() if selected is not None else current_metrics.copy())
            explanation_metrics["trigger_tier"] = tier
            explanation_metrics["keyframe_review_gap"] = float(source_frame - selected.frame_idx) if selected is not None else 0.0
            segment = (
                select_explanation_segment(keyframe_history, selected.frame_idx, source_frame, keyframe_segment_frames)
                if selected is not None
                else []
            )
            segment_written = 0
            for snapshot in segment:
                segment_frame = snapshot.image.copy()
                segment_metrics = snapshot.metrics.copy()
                segment_metrics["trigger_tier"] = tier
                segment_metrics["keyframe_review_gap"] = float(source_frame - snapshot.frame_idx)
                draw_keyframe_explanation(segment_frame, segment_metrics, state)
                writer.write(segment_frame)
                written_frames += 1
                segment_written += 1
            draw_keyframe_explanation(freeze, explanation_metrics, state)
            for _ in range(max(0, tier_pause_frames - segment_written)):
                writer.write(freeze)
                written_frames += 1
            last_keyframe_pause = source_frame
            inserted_keyframes += 1
        elif args.keyframe_pause_policy == "high-value" and tier == 1:
            skipped_minor_keyframes += 1

    cap.release()
    writer.release()
    report = state.summary()
    if not args.disable_goalkeeper_analysis:
        report["goalkeeper_analysis"] = goalkeeper_state.summary()
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote {written_frames} frames at {fps:.2f} fps to {output_video}")
    print(f"Inserted {inserted_keyframes} keyframe pauses")
    print(f"Skipped {skipped_minor_keyframes} ordinary touch candidates")
    print(f"Wrote individual report to {report_path}")
    if wrote_snapshot:
        print(f"Wrote snapshot to {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
