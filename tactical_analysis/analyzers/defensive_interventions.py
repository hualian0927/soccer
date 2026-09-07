"""Conservative defensive-intervention candidates from stable possession changes."""

from __future__ import annotations

from ..base import TacticalAnalyzer, register_analyzer
from ..models import AnalysisContext, AnalyzerOutput, Evidence, Finding, TimelineEvent
from .events import EventTimelineAnalyzer


@register_analyzer("defensive_interventions")
class DefensiveInterventionAnalyzer(TacticalAnalyzer):
    priority = "P1"
    description_zh = "把稳定跨队球权切换整理为可复核的基础防守干预候选"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        base = EventTimelineAnalyzer(self.config).analyze(context)
        changes = [event for event in base.events if event.event_type == "possession_change"]
        events: list[TimelineEvent] = []
        findings: list[Finding] = []
        for index, change in enumerate(changes, 1):
            distance = float(change.metrics.get("ball_displacement_m") or 0.0)
            subtype = "interception_candidate" if distance >= 8.0 else "challenge_regain_candidate"
            subtype_zh = "拦截" if subtype == "interception_candidate" else "对抗夺回"
            confidence = min(0.84, 0.54 + min(distance / 80.0, 0.12) + change.confidence * 0.16)
            metrics = {
                **change.metrics,
                "source_event_id": change.event_id,
                "intervention_subtype": subtype,
                "possession_won": True,
            }
            event = TimelineEvent(
                event_id=f"defensive-intervention-{index:04d}",
                event_type="defensive_intervention_candidate",
                frame=change.frame,
                time_sec=change.time_sec,
                team=change.team,
                actor_track_id=change.target_track_id,
                target_track_id=change.actor_track_id,
                confidence=round(confidence, 4),
                label_zh=f"{subtype_zh}候选",
                metrics=metrics,
                event_subtype=subtype,
                outcome="possession_won",
                start_x=change.start_x,
                start_y=change.start_y,
                end_x=change.end_x,
                end_y=change.end_y,
                evidence_start_sec=max(0.0, change.time_sec - 2.0),
                evidence_end_sec=change.time_sec + 2.5,
                related_event_ids=[change.event_id],
            )
            events.append(event)
            findings.append(
                Finding(
                    finding_id=f"defensive-intervention-{index:04d}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="defensive_intervention_candidate",
                    title_zh=f"{change.time_sec:.2f}秒 {subtype_zh}候选",
                    summary_zh="防守方形成稳定控球，具体动作类型仍需回看原视频确认。",
                    confidence=event.confidence,
                    start_sec=event.evidence_start_sec,
                    end_sec=event.evidence_end_sec,
                    metrics=metrics,
                    evidence=[Evidence(change.frame, change.time_sec, "跨队稳定球权切换及足球位移。", metrics)],
                    limitations_zh=["仅将稳定球权夺回标记为防守干预，不自动判定抢断、犯规或动作规范性。"],
                )
            )
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "stable_possession_change_intervention_v1", "count": len(events)},
            findings=findings,
            events=events,
            warnings=["基础防守干预是可复核候选，动作子类型尚未使用专项标注模型。"],
        )
