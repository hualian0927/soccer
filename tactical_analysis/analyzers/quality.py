"""Input quality checks that gate downstream tactical claims."""

from __future__ import annotations

import statistics

from ..base import TacticalAnalyzer, register_analyzer
from ..models import AnalysisContext, AnalyzerOutput, Evidence, Finding


@register_analyzer("data_quality")
class DataQualityAnalyzer(TacticalAnalyzer):
    priority = "P0"
    description_zh = "检测输入覆盖率、球可见性、球队标签和场地坐标完整性"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        frames = context.frames
        if not frames:
            return AnalyzerOutput(
                analyzer=self.name,
                priority=self.priority,
                warnings=["GSR JSON 中没有可分析帧。"],
            )

        expected_frames = max(1, frames[-1].frame - frames[0].frame + 1)
        frame_coverage = len(frames) / expected_frames
        player_counts = [len(frame.players) for frame in frames]
        calibrated_players = sum(
            player.has_pitch_position for frame in frames for player in frame.players
        )
        total_players = sum(player_counts)
        calibrated_ratio = calibrated_players / max(total_players, 1)
        ball_frames = sum(frame.ball is not None for frame in frames)
        ball_pitch_frames = sum(
            frame.ball is not None and frame.ball.has_pitch_position for frame in frames
        )
        ball_coverage = ball_frames / len(frames)
        ball_pitch_coverage = ball_pitch_frames / len(frames)
        labeled_players = sum(
            player.team in {"left", "right"} for frame in frames for player in frame.players
        )
        team_label_ratio = labeled_players / max(total_players, 1)
        median_players = float(statistics.median(player_counts)) if player_counts else 0.0

        quality_score = (
            0.18 * min(1.0, frame_coverage)
            + 0.27 * min(1.0, calibrated_ratio)
            + 0.28 * min(1.0, ball_pitch_coverage / 0.65)
            + 0.17 * min(1.0, team_label_ratio)
            + 0.10 * min(1.0, median_players / 10.0)
        )
        if quality_score >= 0.78:
            grade, summary = "良好", "数据可支撑基础事件、空间和阵型趋势分析。"
        elif quality_score >= 0.55:
            grade, summary = "可用", "数据可用于候选发现，但关键结论需要结合视频复核。"
        else:
            grade, summary = "偏低", "检测或场地坐标覆盖不足，应限制自动结论强度。"

        warnings: list[str] = []
        if ball_pitch_coverage < 0.35:
            warnings.append("足球场地坐标覆盖较低，传球、射门和球权事件容易漏检。")
        if median_players < 7:
            warnings.append("单帧可见球员偏少，阵型只应解释为局部线型。")
        if team_label_ratio < 0.80:
            warnings.append("球队标签不完整，球权转换和局部人数判断可信度下降。")

        metrics = {
            "quality_score": round(quality_score, 4),
            "grade_zh": grade,
            "observed_frames": len(frames),
            "expected_frame_span": expected_frames,
            "frame_coverage": round(frame_coverage, 4),
            "median_visible_players": round(median_players, 2),
            "player_pitch_coverage": round(calibrated_ratio, 4),
            "team_label_coverage": round(team_label_ratio, 4),
            "ball_frame_coverage": round(ball_coverage, 4),
            "ball_pitch_coverage": round(ball_pitch_coverage, 4),
        }
        finding = Finding(
            finding_id="quality-overview",
            analyzer=self.name,
            priority=self.priority,
            category="data_quality",
            title_zh=f"输入数据质量：{grade}",
            summary_zh=summary,
            confidence=round(quality_score, 4),
            start_sec=frames[0].time_sec,
            end_sec=frames[-1].time_sec,
            metrics=metrics,
            evidence=[
                Evidence(
                    frame=None,
                    time_sec=None,
                    description_zh="质量评分由帧、球、球员、球队标签和场地坐标覆盖率共同计算。",
                    metrics=metrics,
                )
            ],
            limitations_zh=warnings.copy(),
        )
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary=metrics,
            findings=[finding],
            warnings=warnings,
        )
