"""Conservative L1 event candidates for an already projected two-camera clip."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass


L1_GROUPS = {
    "L1-01": "射门与结果",
    "L1-02": "定位球开始与落点",
    "L1-03": "接球、持球与传球",
    "L1-04": "球权转换",
    "L1-05": "防守干预",
    "L1-06": "门将事件",
    "L1-07": "比赛结构",
}


@dataclass(frozen=True)
class Owner:
    person_id: str
    team: str
    distance_m: float


def build_person_index(people: list[dict]) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for person in people:
        if person.get("role") != "Player" or person.get("color") not in {"red", "blue"}:
            continue
        for point in person.get("positions", []):
            by_frame[int(point["frame"])].append({
                "person_id": str(person["person_id"]),
                "team": str(person["color"]),
                "camera": str(person.get("camera") or ""),
                "x_m": float(point["x_m"]),
                "y_m": float(point["y_m"]),
            })
    return by_frame


def nearest_owner(ball: dict, people: list[dict], radius_m: float = 2.6) -> Owner | None:
    if not ball.get("observed") or float(ball.get("confidence") or 0) < 0.4:
        return None
    candidates = []
    for person in people:
        distance = math.hypot(float(ball["x_m"]) - person["x_m"], float(ball["y_m"]) - person["y_m"])
        if distance <= radius_m:
            candidates.append((distance, person))
    if not candidates:
        return None
    source = str(ball.get("source_camera") or "")
    candidates.sort(key=lambda item: item[0] + (0.35 if item[1]["camera"] != source else 0))
    distance, person = candidates[0]
    nearest_opponent = min((value for value, other in candidates if other["team"] != person["team"]),
                           default=float("inf"))
    if nearest_opponent - distance < 0.6:
        return None
    return Owner(person["person_id"], person["team"], round(distance, 3))


def stable_owner_segments(ball_positions: list[dict], people: list[dict], start_sec: float) -> list[dict]:
    ball_by_frame = {int(row["frame"]): row for row in ball_positions}
    people_by_frame = build_person_index(people)
    max_frame = max(ball_by_frame, default=0)
    raw: list[Owner | None] = []
    for frame in range(1, max_frame + 1):
        ball = ball_by_frame.get(frame)
        raw.append(nearest_owner(ball, people_by_frame.get(frame, [])) if ball and float(ball["time_s"]) >= start_sec else None)
    stable: list[Owner | None] = []
    for index, owner in enumerate(raw):
        if owner is None:
            stable.append(None)
            continue
        recent = raw[max(0, index - 4):index + 1]
        votes = sum(item is not None and item.person_id == owner.person_id for item in recent)
        stable.append(owner if votes >= 3 else None)

    segments: list[dict] = []
    for frame, owner in enumerate(stable, 1):
        if owner is None:
            continue
        if segments and segments[-1]["person_id"] == owner.person_id and frame - segments[-1]["end_frame"] <= 6:
            segments[-1]["end_frame"] = frame
            segments[-1]["samples"] += 1
            segments[-1]["distances"].append(owner.distance_m)
        else:
            segments.append({"person_id": owner.person_id, "team": owner.team,
                             "start_frame": frame, "end_frame": frame,
                             "samples": 1, "distances": [owner.distance_m]})
    return [segment for segment in segments if segment["samples"] >= 4]


def path_quality(ball_by_frame: dict[int, dict], start_frame: int, end_frame: int) -> float:
    if end_frame < start_frame:
        return 0.0
    observed = sum(bool(ball_by_frame.get(frame, {}).get("observed"))
                   for frame in range(start_frame, end_frame + 1))
    return observed / (end_frame - start_frame + 1)


def analyze_projection_l1(ball_positions: list[dict], people: list[dict], restarts: list[dict],
                          outside: list[dict], duration: float, fps: float) -> dict:
    ball_by_frame = {int(row["frame"]): row for row in ball_positions}
    active_start = min((float(row["timestamp_sec"]) for row in restarts), default=0.0)
    events: list[dict] = []

    def append(group: str, event_type: str, subtype: str, label: str, time_sec: float,
               start: float, end: float, team: str | None = None,
               actor: str | None = None, recipient: str | None = None,
               confidence: float = 0.5, summary: str = "", attributes: dict | None = None) -> None:
        events.append({
            "event_id": f"projection-l1-{len(events) + 1:03d}",
            "l1_group": group,
            "group_label_zh": L1_GROUPS[group],
            "event_type": event_type,
            "event_subtype": subtype,
            "label_zh": label,
            "timestamp_sec": round(time_sec, 3),
            "frame": int(round(time_sec * fps)) + 1,
            "team_id": team,
            "actor_track_id": actor,
            "recipient_track_id": recipient,
            "outcome": "unknown",
            "confidence": round(confidence, 3),
            "evidence_start_sec": round(max(0.0, start), 3),
            "evidence_end_sec": round(min(duration, end), 3),
            "source": "dual_camera_projection",
            "review_status": "candidate",
            "summary_zh": summary,
            "attributes": attributes or {},
        })

    for restart in restarts:
        append("L1-07", "restart", "kickoff", restart["label_zh"],
               float(restart["timestamp_sec"]), float(restart["evidence_start_sec"]),
               float(restart["evidence_end_sec"]), confidence=0.7,
               summary="中圈长时间静止后连续离开，开球人与真实触球帧待原画面确认。",
               attributes={"evidence": restart["evidence_zh"], "origin": "center_spot"})
    for item in outside:
        time_sec = float(item["start_sec"])
        append("L1-07", "out_of_play", item["line"], item["label_zh"], time_sec,
               time_sec - 4, float(item["end_sec"]) + 15, confidence=0.55,
               summary=f"连续场外投影，{item['possible_restart_zh']}；需要查看最后触球和后续重启。",
               attributes=item)

    segments = stable_owner_segments(ball_positions, people, active_start)
    for index, segment in enumerate(segments):
        start_frame, end_frame = segment["start_frame"], segment["end_frame"]
        start_sec, end_sec = (start_frame - 1) / fps, (end_frame - 1) / fps
        start_ball = ball_by_frame.get(start_frame)
        end_ball = ball_by_frame.get(end_frame)
        if not start_ball or not end_ball:
            continue
        team = segment["team"]
        actor = segment["person_id"]
        observed_ratio = path_quality(ball_by_frame, start_frame, end_frame)
        if observed_ratio < 0.65:
            continue
        append("L1-03", "touch", "stable_proximity", "近球控制候选", start_sec,
               start_sec - 1.2, end_sec + 1.2, team, actor, confidence=0.48,
               summary=f"{'红队' if team == 'red' else '蓝队'}球员与足球连续接近 {segment['samples']} 帧；尚未确认脚部实际触球。",
               attributes={"proximity_samples": segment["samples"],
                           "median_distance_m": round(sorted(segment["distances"])[len(segment["distances"]) // 2], 2)})
        distance = math.hypot(float(end_ball["x_m"]) - float(start_ball["x_m"]),
                              float(end_ball["y_m"]) - float(start_ball["y_m"]))
        if end_sec - start_sec >= 0.8 and distance >= 5.0 and observed_ratio >= 0.75:
            append("L1-03", "carry", "unverified_controlled_movement", "持球移动候选", start_sec,
                   start_sec - 1, end_sec + 1, team, actor, confidence=0.48,
                   summary=f"近球球员持续移动，足球起止位移约 {distance:.1f} 米；控球与过人待原画面确认。",
                   attributes={"ball_displacement_m": round(distance, 2), "duration_sec": round(end_sec - start_sec, 2)})

        if index == 0:
            continue
        previous = segments[index - 1]
        if previous["person_id"] == actor:
            continue
        if previous["person_id"].split("#", 1)[0] != actor.split("#", 1)[0]:
            continue
        gap_sec = (start_frame - previous["end_frame"]) / fps
        if not 0 <= gap_sec <= 2.0:
            continue
        old_ball = ball_by_frame.get(previous["end_frame"])
        if not old_ball or path_quality(ball_by_frame, previous["end_frame"], start_frame) < 0.7:
            continue
        displacement = math.hypot(float(start_ball["x_m"]) - float(old_ball["x_m"]),
                                  float(start_ball["y_m"]) - float(old_ball["y_m"]))
        if displacement < 4.0:
            continue
        transfer_start = (previous["end_frame"] - 1) / fps
        if previous["team"] == team:
            append("L1-03", "pass", "same_team_transfer", "传球候选", start_sec,
                   transfer_start - 1.5, start_sec + 1.5, team, previous["person_id"], actor,
                   confidence=0.53,
                   summary=f"同队近球球员切换，足球位移约 {displacement:.1f} 米；传球动作和成功结果待视频复核。",
                   attributes={"ball_displacement_m": round(displacement, 2), "transfer_gap_sec": round(gap_sec, 2)})
            append("L1-03", "receive", "stable_proximity_after_transfer", "接球候选", start_sec,
                   start_sec - 1.2, start_sec + 1.5, team, actor, confidence=0.48,
                   summary="同队转移后新球员持续接近足球；真实接球动作待视频复核。")
        else:
            append("L1-04", "possession_change", "cross_team_proximity_change", "球权转换候选", start_sec,
                   transfer_start - 1.5, start_sec + 2, team, previous["person_id"], actor,
                   confidence=0.5,
                   summary="足球近邻由一队变为另一队；争抢、反弹和真实夺回需原画面确认。",
                   attributes={"from_team": previous["team"], "to_team": team,
                               "ball_displacement_m": round(displacement, 2)})

    events.sort(key=lambda item: (item["timestamp_sec"], item["event_id"]))
    counts = Counter(item["l1_group"] for item in events)
    return {
        "schema_version": "projection-l1-v1",
        "scope": "evidence_first_candidates_not_official_match_events",
        "summary": {
            "total_candidates": len(events),
            "group_counts": {group: counts[group] for group in L1_GROUPS},
            "stable_near_ball_segments": len(segments),
            "active_play_gate_sec": round(active_start, 2),
            "confirmed_events": 0,
        },
        "events": events,
        "unsupported_zh": {
            "L1-01": "射门与结果需要踢球动作、球门结果或连续视频确认。",
            "L1-02": "本片没有可信的角球、任意球等定位球证据；中圈开球属于比赛结构中的重启候选。",
            "L1-05": "抢断、拦截和解围无法仅由最近球员切换确认。",
            "L1-06": "门将扑救需要门将动作与是否触球的视觉证据。",
            "cross_camera": "不同机位的 track ID 不作传球人或接球人关联；本版仅同机位稳定近球切换生成传球候选。",
        },
    }
