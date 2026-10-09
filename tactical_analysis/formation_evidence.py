"""Frame-anchored formation evidence without inventing off-screen players."""

from bisect import bisect_right
from collections import Counter
import math

from .analyzers.formation import estimate_frame


def scene_at(time, cuts):
    return bisect_right(cuts, time)


def frame_distribution(frame, team):
    estimate = estimate_frame(frame, team, 6)
    if not estimate:
        return None
    indexed = {p.track_id:p for p in frame.players if p.team == team and p.role == "player"}
    lines = []
    for ids in estimate["line_members"]:
        members = []
        for track_id in ids:
            p = indexed[track_id]
            box = p.bbox or {}
            if not all(k in box and math.isfinite(box[k]) for k in ("x","y","w","h")) or box["w"] <= 0 or box["h"] <= 0:
                continue
            members.append({"id":track_id,"box":box,"pitch":[p.pitch_x,p.pitch_y]})
        lines.append(sorted(members,key=lambda p:p["pitch"][1]))
    if sum(map(len,lines)) != estimate["visible_players"]:
        return None
    keepers = [p for p in frame.players if p.team == team and p.role == "goalkeeper"
               and p.pitch_x is not None and math.isfinite(p.pitch_x) and abs(p.pitch_x) > 20]
    return {**estimate,"lines":lines,"directionObserved":len(keepers) == 1}


def describe_window(frames, team, time, cuts, radius=2.0):
    scene = scene_at(time,cuts)
    samples = [frame_distribution(f,team) for f in frames
               if abs(f.time_sec-time) <= radius and scene_at(f.time_sec,cuts) == scene]
    valid = [x for x in samples if x]
    if not valid:
        return None
    counts = Counter(tuple(x["line_counts"]) for x in valid)
    shape,support = counts.most_common(1)[0]
    matching = [x for x in valid if tuple(x["line_counts"]) == shape]
    representative = min(matching,key=lambda x:abs(x["time_sec"]-time))
    temporal_ratio = support / max(1,len(samples))
    stable = (support >= 3 and temporal_ratio >= .65
              and max(x["time_sec"] for x in matching)-min(x["time_sec"] for x in matching) >= 1)
    geometry = (representative["line_separation_m"] >= 5 and representative["width_m"] >= 15
                and representative["depth_m"] >= 20)
    complete = representative["complete_visible_team"]
    exact_template = "-".join(map(str,shape)) == representative["template_candidate"]
    template_supported = stable and geometry and complete and exact_template and all(x["directionObserved"] for x in matching)
    title = f"{representative['template_candidate']} 阵型候选" if template_supported else "可见分线 " + " / ".join(map(str,shape))
    return {"title":title,"sceneId":scene,"representative":representative,
            "templateSupported":template_supported,"stable":stable,"sampleCount":len(samples),
            "supportingSamples":support,"supportRatio":round(temporal_ratio,3),
            "status":"同镜头结构候选" if template_supported else "局部站位，不确认完整阵型",
            "calibrationVerified":False,
            "limitations":["连线按已有球场投影分组，球队身份与标定仍需独立核验。",
                           "仅连接真实检测到的球员；没有补画画面外球员。",
                           "短时站位不等于整场固定阵型，支持比例不是识别准确率。"]}
