"""Defensive-third risk by dangerous opponent entries and shots."""

from __future__ import annotations

from collections import Counter, defaultdict

from ..base import TacticalAnalyzer, register_analyzer
from ..models import HALF_WIDTH, AnalysisContext, AnalyzerOutput, Evidence, Finding, TimelineEvent
from .events import EventTimelineAnalyzer
from .progression import extract_pass_records


def risk_channel(y: float) -> str:
    if y > HALF_WIDTH / 3.0:
        return "upper_flank"
    if y < -HALF_WIDTH / 3.0:
        return "lower_flank"
    return "central"


CHANNEL_LABELS = {"upper_flank": "上方边路", "central": "中路", "lower_flank": "下方边路"}


@register_analyzer("defensive_third_risk")
class DefensiveThirdRiskAnalyzer(TacticalAnalyzer):
    priority = "P2"
    description_zh = "按对手推进进入、穿线、传中和射门聚合防守三区危险来源"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        event_output = EventTimelineAnalyzer(self.config).analyze(context)
        shots = [event for event in event_output.events if event.event_type == "shot_candidate"]
        carries = [event for event in event_output.events if event.event_type == "carry_candidate"]
        passes = extract_pass_records(context, self.config)
        risks: list[dict[str, object]] = []

        for record in passes:
            flags = {
                "final_third_entry": bool(record["final_third_entry"]),
                "penalty_area_entry": bool(record["penalty_area_entry"]),
                "line_break": int(record["lines_broken"]) > 0,
                "cross": bool(record["cross_candidate"]),
                "through_ball": bool(record["through_ball_candidate"]),
            }
            if not any(flags.values()):
                continue
            score = 1.0
            score += 1.2 * flags["penalty_area_entry"] + 0.8 * flags["line_break"]
            score += 0.7 * flags["through_ball"] + 0.5 * flags["cross"]
            attacking_team = str(record["team"])
            risks.append(
                {
                    "time_sec": record["time_sec"],
                    "frame": record["frame"],
                    "attacking_team": attacking_team,
                    "defending_team": "right" if attacking_team == "left" else "left",
                    "channel": risk_channel(float(record["end_y"])),
                    "source": "dangerous_pass",
                    "risk_score": round(score, 3),
                    "actor_track_id": record["actor_track_id"],
                    "target_track_id": record["target_track_id"],
                    **flags,
                    "end_x": record["end_x"],
                    "end_y": record["end_y"],
                }
            )
        for shot in shots:
            if shot.team not in {"left", "right"}:
                continue
            y = float(shot.metrics.get("shot_y", 0.0))
            xg = float(shot.metrics.get("xg_context_proxy", shot.metrics.get("xg_lite", 0.0)))
            risks.append(
                {
                    "time_sec": shot.time_sec,
                    "frame": shot.frame,
                    "attacking_team": shot.team,
                    "defending_team": "right" if shot.team == "left" else "left",
                    "channel": risk_channel(y),
                    "source": "shot",
                    "risk_score": round(3.0 + 3.0 * xg, 3),
                    "actor_track_id": shot.actor_track_id,
                    "target_track_id": None,
                    "shot_xg_context_proxy": xg,
                    "shot_distance_m": shot.metrics.get("shot_distance_m"),
                }
            )
        for carry in carries:
            if carry.team not in {"left", "right"}:
                continue
            end_x = float(carry.metrics.get("end_x", 0.0))
            direction = 1 if carry.team == "left" else -1
            if end_x * direction < 17.5:
                continue
            end_y = float(carry.metrics.get("end_y", 0.0))
            risks.append(
                {
                    "time_sec": carry.time_sec,
                    "frame": carry.frame,
                    "attacking_team": carry.team,
                    "defending_team": "right" if carry.team == "left" else "left",
                    "channel": risk_channel(end_y),
                    "source": "carry_entry",
                    "risk_score": 1.4,
                    "actor_track_id": carry.actor_track_id,
                    "target_track_id": None,
                    "end_x": end_x,
                    "end_y": end_y,
                }
            )

        timeline_events: list[TimelineEvent] = []
        for risk in sorted(risks, key=lambda item: float(item["risk_score"]), reverse=True):
            if float(risk["risk_score"]) < 2.2:
                continue
            timeline_events.append(
                TimelineEvent(
                    event_id=f"defensive-risk-{len(timeline_events) + 1:03d}",
                    event_type="defensive_risk_candidate",
                    frame=int(risk["frame"]),
                    time_sec=float(risk["time_sec"]),
                    team=str(risk["defending_team"]),
                    actor_track_id=risk["actor_track_id"],
                    target_track_id=risk["target_track_id"],
                    confidence=round(min(0.9, 0.45 + float(risk["risk_score"]) / 10.0), 4),
                    label_zh=f"{CHANNEL_LABELS[str(risk['channel'])]}防守风险候选",
                    metrics=risk,
                )
            )

        findings: list[Finding] = []
        team_summary = {}
        for team in ("left", "right"):
            team_risks = [item for item in risks if item["defending_team"] == team]
            channel_scores: defaultdict[str, float] = defaultdict(float)
            source_counts = Counter()
            for risk in team_risks:
                channel_scores[str(risk["channel"])] += float(risk["risk_score"])
                source_counts[str(risk["source"])] += 1
            top_channel = max(channel_scores, key=channel_scores.get) if channel_scores else None
            team_summary[team] = {
                "risk_events": len(team_risks),
                "channel_risk_scores": {key: round(value, 3) for key, value in channel_scores.items()},
                "source_counts": dict(source_counts),
                "highest_risk_channel": top_channel,
            }
            if top_channel is None:
                continue
            representative = max(
                (item for item in team_risks if item["channel"] == top_channel),
                key=lambda item: float(item["risk_score"]),
            )
            total_score = sum(channel_scores.values())
            share = channel_scores[top_channel] / max(total_score, 1e-6)
            findings.append(
                Finding(
                    finding_id=f"defensive-risk-{team}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="defensive_third_risk",
                    title_zh=f"{'左队' if team == 'left' else '右队'}主要风险来自{CHANNEL_LABELS[top_channel]}",
                    summary_zh=(
                        f"该通道累计风险分{channel_scores[top_channel]:.1f}，占全部可见防守风险{share:.0%}；"
                        f"样本由对手进入、穿线、传中、持球推进和射门共同构成。"
                    ),
                    confidence=round(min(0.88, 0.5 + min(len(team_risks), 10) * 0.025 + share * 0.15), 4),
                    start_sec=max(0.0, float(representative["time_sec"]) - 4.0),
                    end_sec=float(representative["time_sec"]) + 4.0,
                    metrics={**team_summary[team], "representative_risk": representative},
                    evidence=[Evidence(int(representative["frame"]), float(representative["time_sec"]), "该防守通道中风险分最高的代表事件。", representative)],
                    limitations_zh=["风险分是相对排序代理，不等同于失球概率；转播画面外的防守球员无法计入。"],
                )
            )
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "dangerous_entry_channel_risk_proxy_v1", "teams": team_summary, "risk_events": risks},
            findings=findings,
            events=timeline_events,
            warnings=["防守三区风险用于定位复核素材，不是失球因果归因或球员责任评分。"],
        )
