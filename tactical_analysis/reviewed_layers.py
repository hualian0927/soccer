"""Evidence-gated organization rules on reviewed events, plus spatial coverage checks."""

from collections import Counter
import math


def accepted(event):
    return (event.get("publishedForStatistics") is True
            and event.get("reviewStatus") != "rejected"
            and event.get("visionStatus") == "visual_supported_candidate")


def organization_candidates(events, maximum_gap=12.0):
    """Link sparse evidence; never equate linked nodes with uninterrupted possession."""
    ordered = sorted(events, key=lambda item: item["time"])
    output = []
    for index, current in enumerate(ordered):
        if not accepted(current) or current.get("team") not in {"left", "right"}:
            continue
        if current.get("subtype") == "carry":
            output.append(_sequence("progression", [current], "持球推进与防守回撤"))
        if index == 0:
            continue
        previous = ordered[index - 1]
        # Unknown/rejected intervening nodes break a sequence, as do known cuts or restarts.
        if not accepted(previous) or previous.get("team") != current["team"]:
            continue
        if current["time"] - previous["time"] > maximum_gap:
            continue
        if previous.get("sceneId") is None or current.get("sceneId") is None or previous["sceneId"] != current["sceneId"]:
            continue
        if previous.get("category") in {"出界", "定位球"} or current.get("category") in {"出界", "定位球"}:
            continue
        if previous.get("subtype") == "pass" and current.get("subtype") == "cross":
            output.append(_sequence("wide_delivery", [previous, current], "短传拉边后的传中组织"))
        elif previous.get("subtype") == current.get("subtype") == "foot_pass" and all(
                event.get("category") == "门将事件" for event in (previous, current)):
            output.append(_sequence("keeper_recycle", [previous, current], "门将参与后场循环出球"))
    return output


def _sequence(kind, events, title):
    return {"id": f"organization-{kind}-{events[0]['id']}", "kind": kind, "title": title,
            "time": events[0]["time"], "end": events[-1]["time"], "team": events[0]["team"],
            "sourceEventIds": [e["id"] for e in events], "observedNodeCount": len(events),
            "status": "规则候选，需连续画面复核", "completePossession": False,
            "summary": "按同队事件类型与时间邻接生成，不推断未观察到的传球次数或射门结果。",
            "limitations": ["抽样事件不能用于计算全场控球时长、PPDA或完整进攻成功率。"]}


def spatial_coverage(context, time, radius=1.0, minimum_players=6):
    """Screen visibility before displaying metric claims; not a calibration accuracy test."""
    frames = [frame for frame in context.frames if abs(frame.time_sec - time) <= radius]
    def valid(observation):
        return (observation.pitch_x is not None and observation.pitch_y is not None
                and math.isfinite(observation.pitch_x) and math.isfinite(observation.pitch_y)
                and abs(observation.pitch_x) <= 52.5 and abs(observation.pitch_y) <= 34)
    counts = []
    for frame in frames:
        teams = {team: len({p.track_id for p in frame.observations
                           if p.team == team and p.role == "player" and valid(p)}) for team in ("left", "right")}
        counts.append((teams, any(p.role == "ball" and valid(p) for p in frame.observations)))
    coverage = len(frames) / max(1, round(2 * radius * context.fps))
    usable = sum(ball and min(teams.values()) >= minimum_players for teams, ball in counts)
    ratio = usable / max(1, len(frames))
    passed = coverage >= 0.5 and ratio >= 0.7
    return {"passed": passed, "sampleCount": len(frames), "frameCoverage": round(min(1, coverage), 3),
            "usableRatio": round(ratio, 3), "minimumPlayersPerTeam": minimum_players,
            "reason": "仅通过可见性门槛，仍需标定与画面复核" if passed else "球或双队可见人数不足，隐藏精确空间结论"}


def reviewed_statistics(events):
    subset = [e for e in events if accepted(e)]
    return {"scope": "本次复核节点，不是全场统计", "acceptedCount": len(subset),
            "subtypes": dict(Counter(e.get("subtype", "unknown") for e in subset)),
            "unknownTeamCount": sum(e.get("team") not in {"left", "right"} for e in subset)}
