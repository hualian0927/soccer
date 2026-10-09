"""Shot-local identities and conservative, image-coordinate player summaries."""

import math
from collections import Counter


class LocalIdentities:
    def __init__(self):
        self.next_id = 1
        self.mapping = {}
        self.anchors = {}

    def reset_scene(self):
        self.mapping.clear()
        self.anchors.clear()

    def link(self, tracker_id, source_id, box, frame, fps, used, edge=False):
        """Reconnect short fragments only with source, time and spatial agreement."""
        old = self.anchors.get(source_id) if source_id is not None else None
        if old:
            old_id, old_box, old_frame, old_edge = old
            distance = math.hypot((box[0]+box[2]-old_box[0]-old_box[2])/2,
                                  (box[1]+box[3]-old_box[1]-old_box[3])/2)
            if (0 < frame-old_frame <= fps and not old_edge and old_id not in used
                    and distance < 1.25*max(box[3]-box[1], old_box[3]-old_box[1])):
                self.mapping[tracker_id] = old_id
        if self.mapping.get(tracker_id) in used:
            self.mapping.pop(tracker_id, None)
        local_id = self.assign(tracker_id)
        if source_id is not None:
            self.anchors[source_id] = (local_id, box, frame, edge)
        return local_id

    def assign(self, tracker_id):
        if tracker_id not in self.mapping:
            self.mapping[tracker_id] = self.next_id
            self.next_id += 1
        return self.mapping[tracker_id]


def summarize_selection(data, player_id, start, end):
    if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= data["duration"]:
        raise ValueError("Invalid time range")
    if end - start > 30:
        raise ValueError("Select at most 30 seconds in one shot")
    fps = data["fps"]
    frames = [f for f in data["frames"] if start <= f["frame"] / fps < end]
    if len({f["scene"] for f in frames}) > 1:
        raise ValueError("Selection crosses a camera cut; select one shot")
    samples = [(f, p) for f in frames for p in f["players"] if p["id"] == player_id]
    if not samples:
        raise ValueError("Selected player is not visible in this interval")
    observed = [(f, p) for f, p in samples if not p.get("predicted", False)]
    first, last = samples[0][1]["box"], samples[-1][1]["box"]
    near_ball = 0
    ball_frames = 0
    for frame, player in observed:
        if not frame.get("balls"):
            continue
        ball_frames += 1
        x, y, w, h = player["box"]
        if any(math.hypot((b[0] + b[2] / 2 - x - w / 2) * data["width"],
                          (b[1] + b[3] / 2 - y - h) * data["height"]) < max(12, .6 * h * data["height"])
               for b in frame["balls"]):
            near_ball += 1
    expected = max(1, round((end - start) * fps))
    return {"player_id": player_id, "start": start, "end": end, "scene": samples[0][0]["scene"],
        "observed_seconds": round(len(observed) / fps, 2),
        "coverage": round(min(1, len(observed) / expected), 3),
        "ball_observed_frames": ball_frames, "ball_near_feet_frames": near_ball,
        "screen_displacement_pct": [round(100 * (last[i] - first[i]), 2) for i in (0, 1)],
        "team": Counter(p["team"] for _, p in observed).most_common(1)[0][0] if observed else "unknown",
        "status": "pending_visual_review",
        "limitations": ["画面位移包含相机运动，不是实际跑动距离或速度。",
                         "脚边球框接近只表示图像邻近，不等于触球、控球或成功传球。",
                         "同队遮挡、漏检或快速交叉仍可能换号；临时编号不是球衣号码。"]}
