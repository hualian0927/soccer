"""Shot-ending attacking chains assembled from possession and action events."""

from __future__ import annotations

from collections import Counter

from ..base import TacticalAnalyzer, register_analyzer
from ..models import AnalysisContext, AnalyzerOutput, Evidence, Finding, TimelineEvent
from .events import EventTimelineAnalyzer
from .progression import extract_pass_records


@register_analyzer("attacking_chains")
class AttackingChainAnalyzer(TacticalAnalyzer):
    priority = "P2"
    description_zh = "回溯射门前的传球、推进和区域进入，解释进攻如何形成"

    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        event_output = EventTimelineAnalyzer(self.config).analyze(context)
        events = sorted(event_output.events, key=lambda item: (item.time_sec, item.event_id))
        passes = extract_pass_records(context, self.config)
        lookback_sec = float(self.config.get("maximum_chain_lookback_sec", 20.0))
        shots = [event for event in events if event.event_type == "shot_candidate"]
        carries = [event for event in events if event.event_type == "carry_candidate"]
        changes = [event for event in events if event.event_type == "possession_change"]
        corners = [event for event in events if event.event_type == "corner_candidate"]
        chains: list[dict[str, object]] = []
        timeline_events: list[TimelineEvent] = []
        findings: list[Finding] = []

        for shot in shots:
            if shot.team not in {"left", "right"}:
                continue
            preceding_changes = [event for event in changes if event.time_sec <= shot.time_sec]
            last_change = preceding_changes[-1] if preceding_changes else None
            start_sec = max(0.0, shot.time_sec - lookback_sec)
            if last_change and last_change.team == shot.team:
                start_sec = max(start_sec, last_change.time_sec)
            elif last_change and last_change.team != shot.team:
                start_sec = max(start_sec, last_change.time_sec)
            chain_passes = [record for record in passes if record["team"] == shot.team and start_sec <= record["time_sec"] <= shot.time_sec]
            chain_carries = [event for event in carries if event.team == shot.team and start_sec <= event.time_sec <= shot.time_sec]
            chain_corners = [event for event in corners if event.team == shot.team and start_sec <= event.time_sec <= shot.time_sec]
            actions = [
                {"time_sec": record["time_sec"], "type": record["pass_type"], "actor_track_id": record["actor_track_id"], "target_track_id": record["target_track_id"]}
                for record in chain_passes
            ]
            actions.extend(
                {"time_sec": event.time_sec, "type": "carry", "actor_track_id": event.actor_track_id, "target_track_id": None}
                for event in chain_carries
            )
            actions.extend(
                {"time_sec": event.time_sec, "type": "corner", "actor_track_id": event.actor_track_id, "target_track_id": None}
                for event in chain_corners
            )
            actions.append({"time_sec": shot.time_sec, "type": "shot", "actor_track_id": shot.actor_track_id, "target_track_id": None})
            actions.sort(key=lambda item: float(item["time_sec"]))
            if actions:
                start_sec = float(actions[0]["time_sec"])
            duration = shot.time_sec - start_sec
            if chain_corners:
                route = "set_piece_attack"
                route_label = "定位球进攻"
            elif any(record["cross_candidate"] for record in chain_passes):
                route = "wide_cross_attack"
                route_label = "边路传中进攻"
            elif any(record["through_ball_candidate"] for record in chain_passes):
                route = "central_penetration"
                route_label = "中路穿透进攻"
            elif len(chain_passes) >= 4:
                route = "combination_attack"
                route_label = "连续配合进攻"
            elif duration <= 8.0 or len(chain_passes) <= 1:
                route = "direct_attack"
                route_label = "快速直接进攻"
            else:
                route = "mixed_attack"
                route_label = "混合推进进攻"
            metrics = {
                "team": shot.team,
                "start_sec": round(start_sec, 3),
                "shot_sec": shot.time_sec,
                "duration_sec": round(duration, 3),
                "route": route,
                "action_count": len(actions),
                "pass_count": len(chain_passes),
                "carry_count": len(chain_carries),
                "progressive_pass_count": sum(bool(record["progressive"]) for record in chain_passes),
                "line_breaking_pass_count": sum(int(record["lines_broken"] > 0) for record in chain_passes),
                "final_third_entry_count": sum(bool(record["final_third_entry"]) for record in chain_passes),
                "penalty_area_entry_count": sum(bool(record["penalty_area_entry"]) for record in chain_passes),
                "total_forward_progress_m": round(sum(max(0.0, float(record["forward_progress_m"])) for record in chain_passes), 3),
                "pass_type_counts": dict(Counter(str(record["pass_type"]) for record in chain_passes)),
                "shot_actor_track_id": shot.actor_track_id,
                "shot_xg_context_proxy": shot.metrics.get("xg_context_proxy"),
                "actions": actions,
            }
            confidence = min(0.9, 0.5 + min(len(actions), 6) * 0.04 + (0.08 if chain_passes else 0.0))
            chains.append(metrics)
            timeline_event = TimelineEvent(
                event_id=f"attack-chain-{len(chains):03d}",
                event_type="attacking_chain_candidate",
                frame=shot.frame,
                time_sec=shot.time_sec,
                team=shot.team,
                actor_track_id=shot.actor_track_id,
                target_track_id=None,
                confidence=round(confidence, 4),
                label_zh=f"{route_label}形成射门候选",
                metrics=metrics,
            )
            timeline_events.append(timeline_event)
            findings.append(
                Finding(
                    finding_id=f"attacking-chain-{len(chains):03d}",
                    analyzer=self.name,
                    priority=self.priority,
                    category="attacking_chain_candidate",
                    title_zh=f"{shot.time_sec:.2f}秒 {'左队' if shot.team == 'left' else '右队'}{route_label}",
                    summary_zh=(
                        f"进攻链持续{duration:.1f}秒，包含{len(chain_passes)}次传球、{len(chain_carries)}次持球推进，"
                        f"累计向前推进约{float(metrics['total_forward_progress_m']):.1f}米后形成射门。"
                    ),
                    confidence=timeline_event.confidence,
                    start_sec=max(0.0, start_sec - 2.0),
                    end_sec=shot.time_sec + 4.0,
                    metrics=metrics,
                    evidence=[Evidence(shot.frame, shot.time_sec, "从射门向前回溯同队动作与最近球权边界。", metrics)],
                    limitations_zh=["切镜、足球漏检和跟踪 ID 变化会截断进攻链；链路只代表可见动作。"],
                )
            )

        team_summary = {}
        for team in ("left", "right"):
            items = [item for item in chains if item["team"] == team]
            team_summary[team] = {
                "shot_ending_chains": len(items),
                "route_counts": dict(Counter(str(item["route"]) for item in items)),
                "mean_chain_duration_sec": round(sum(float(item["duration_sec"]) for item in items) / len(items), 3) if items else None,
                "mean_actions_per_chain": round(sum(int(item["action_count"]) for item in items) / len(items), 3) if items else None,
            }
        return AnalyzerOutput(
            analyzer=self.name,
            priority=self.priority,
            summary={"method": "shot_backward_visible_action_chain_v1", "teams": team_summary, "chains": chains},
            findings=findings,
            events=timeline_events,
            warnings=["进攻链是可见事件回溯，不包含越位、犯规、比赛中断或足球漏检期间的隐藏动作。"],
        )
