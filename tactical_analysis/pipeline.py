"""Pipeline orchestration for registered tactical analyzers."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import analyzers as _built_in_analyzers  # noqa: F401
from .base import ANALYZER_REGISTRY
from .gsr_io import infer_video_fps, load_gsr_context
from .models import AnalysisReport, AnalyzerOutput


DEFAULT_CONFIG: dict[str, Any] = {
    "pipeline": {
        "analyzers": [
            "data_quality",
            "event_timeline",
            "defensive_interventions",
            "spatial_structure",
            "team_shape_engine",
            "formation_tendency",
            "relational_space",
            "set_piece_delivery",
            "goalkeeper_interventions",
            "phase_of_play",
            "progression_analysis",
            "attacking_chains",
            "defensive_third_risk",
            "pass_network",
            "numerical_superiority",
            "pressing_analysis",
            "defensive_gaps",
            "transition_analysis",
        ],
        "continue_on_error": True,
    },
    "analyzers": {},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return deepcopy(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        custom = yaml.safe_load(handle) or {}
    return deep_merge(DEFAULT_CONFIG, custom)


class TacticalAnalysisPipeline:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = deep_merge(DEFAULT_CONFIG, config or {})

    @classmethod
    def from_config_file(cls, config_path: Path | None) -> "TacticalAnalysisPipeline":
        return cls(load_config(config_path))

    def run(
        self,
        json_path: Path,
        video_path: Path | None = None,
        fps: float | None = None,
        selected_analyzers: list[str] | None = None,
    ) -> AnalysisReport:
        resolved_fps = fps or infer_video_fps(video_path, fallback=25.0)
        context = load_gsr_context(json_path=json_path, fps=resolved_fps, video_path=video_path)
        names = selected_analyzers or list(self.config["pipeline"]["analyzers"])
        unknown = [name for name in names if name not in ANALYZER_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown analyzers: {', '.join(unknown)}")

        outputs: list[AnalyzerOutput] = []
        continue_on_error = bool(self.config["pipeline"].get("continue_on_error", True))
        for name in names:
            analyzer_config = self.config.get("analyzers", {}).get(name, {})
            analyzer = ANALYZER_REGISTRY[name](analyzer_config)
            try:
                outputs.append(analyzer.analyze(context))
            except Exception as exc:
                if not continue_on_error:
                    raise
                outputs.append(
                    AnalyzerOutput(
                        analyzer=name,
                        priority=analyzer.priority,
                        warnings=[f"分析器运行失败：{type(exc).__name__}: {exc}"],
                    )
                )

        return AnalysisReport(
            schema_version="1.5.0",
            generated_at=datetime.now(timezone.utc).isoformat(),
            source={
                "gsr_json": str(json_path),
                "video": str(video_path) if video_path else None,
                "fps": resolved_fps,
                "duration_sec": round(context.duration_sec, 3),
                **context.metadata,
            },
            config=self.config,
            analyzer_outputs=outputs,
            provenance={
                "input_adapter": "SoccerNetGSR",
                "analysis_style": "evidence_first_candidate_analysis",
                "visual_scope": "video_tracking_only_no_gps_or_wearables",
                "paper_inspired_components": {
                    "SoccerMaster": "统一多任务数据契约与共享比赛上下文",
                    "PathCRF": "带逻辑约束的稳定球权路径和事件切换",
                    "TacticAI": "球员关系图、局部人数和定位球结构表示",
                    "BASKET": "跨帧证据聚合后再输出技能或战术结论",
                    "Monte Carlo Pass Search": "预留反事实传球评估接口，基础版不伪造世界模型结果",
                    "Synthetic Data": "报告中保留来源、用途边界和限制说明",
                    "Skor-xG": "射门特征接口加入防守压力与门将站位，当前仅输出未校准代理值",
                    "Passing Networks": "由视觉传球候选构建有向加权图、中心性、连通性与传球熵",
                    "FIFA Phase of Play": "用稳定球权段组织比赛阶段、五秒转换窗口和压迫回合",
                    "FIFA Team Shape": "按有球/无球状态聚合球队宽度、纵深、三线高度、线距和紧凑度",
                    "Priority P0-P2 product chain": "事件、射门结构、区域、定位球、门将、进攻链路与防守风险统一进入交付层",
                    "P3 advanced candidates": "以持续时序窗口组织动态多打少、高位逼抢、线间空档和攻防转换速度",
                },
                "compatible_renderers": [
                    "make_tactical_visualization_video.py",
                    "make_tactical_report_video.py",
                    "make_formation_analysis_video.py",
                    "make_offensive_analysis_video.py",
                    "make_individual_technique_analysis_video.py",
                    "export_tactical_highlights.py",
                ],
            },
            limitations_zh=[
                "基础框架输出的是可复核候选和趋势，不把启发式结果包装成职业数据商真值。",
                "单目转播视频存在遮挡、切镜和局部视野，阵型与区域统计必须结合数据质量阅读。",
                "射门、传球和球权依赖足球轨迹；足球漏检时，事件召回率会明显下降。",
                "个人动作规范性仍应由近景姿态模块和教练复核，不能从远景 GSR 直接推断。",
                "本框架仅使用视频视觉信息，不读取 GPS、可穿戴设备、生理信号或训练负荷。",
            ],
        )
