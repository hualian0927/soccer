"""Build full-video shot-local player IDs using BoT-SORT and cached GSR detections."""

import argparse
import json
import subprocess
import shutil
import sys
from collections import defaultdict, Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from tactical_analysis.gsr_io import parse_frame
from tactical_analysis.player_focus import LocalIdentities
from tactical_analysis.player_tracker import JerseyBOTSORT, shirt_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gsr-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = defaultdict(list)
    for p in json.loads(args.gsr_json.read_text())["predictions"]:
        rows[parse_frame(p["image_id"]) - 1].append(p)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0:
        raise ValueError("Cannot decode input video")
    if args.max_seconds:
        total = min(total, round(args.max_seconds * fps))
    # Reuse existing weights as an appearance encoder, not a jersey/identity classifier.
    weights = str(Path(__file__).resolve().parent / "yolov8n.pt")
    fallback = YOLO(weights)
    tracker = JerseyBOTSORT(SimpleNamespace(track_high_thresh=.05, track_low_thresh=.01,
        new_track_thresh=.35, track_buffer=round(fps), match_thresh=.8, fuse_score=True,
        gmc_method="sparseOptFlow", proximity_thresh=.4, appearance_thresh=.8,
        with_reid=True, model=weights), frame_rate=round(fps))
    tracker.encoder.model.predictor.args.imgsz = 128
    identities, frames, scene, previous = LocalIdentities(), [], 0, None
    cuts, previous_boxes = [], {}
    writer = None
    temporary = args.output_dir / "numbered_silent.mp4"
    if args.render:
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    votes = defaultdict(Counter)
    for index in range(total):
        ok, image = cap.read()
        if not ok:
            break
        small = cv2.resize(image, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Hard cuts reset identity; camera pans are handled by BoT-SORT GMC.
        if previous is not None:
            delta = np.mean(np.abs(gray.astype(float) - previous.astype(float)))
            corr = cv2.matchTemplate(gray, previous, cv2.TM_CCOEFF_NORMED)[0, 0]
            if delta > 28 and corr < .55:
                tracker.reset()
                identities.reset_scene()
                previous_boxes.clear()
                scene += 1
                cuts.append(index / fps)
        previous = gray
        detections, labels, balls = [], [], []
        for p in rows[index]:
            b = p.get("bbox_image") or {}
            if not all(k in b and np.isfinite(b[k]) for k in ("x", "y", "w", "h")):
                continue
            x, y, w, h = (float(b[k]) for k in ("x", "y", "w", "h"))
            x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            attr = p.get("attributes", {})
            if attr.get("role") == "ball":
                balls.append([x1 / width, y1 / height, (x2-x1) / width, (y2-y1) / height])
            elif attr.get("role") in {"player", "goalkeeper", "referee"}:
                detections.append([x1, y1, x2, y2, float(p.get("score") or .8), 0])
                labels.append((attr.get("team") or "unknown", attr.get("role"), p.get("source_track_id",p.get("track_id"))))
        if len(detections) < 3:
            result = fallback.predict(image, classes=[0], conf=.25, imgsz=1280, verbose=False)[0]
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, _ = box
                if not any(abs((d[0]+d[2]-x1-x2)/2) < (x2-x1)*.5 and
                           abs((d[1]+d[3]-y1-y2)/2) < (y2-y1)*.5 for d in detections):
                    detections.append([x1,y1,x2,y2,conf,0])
                    labels.append(("unknown", "person", None))
        for detection in detections:
            detection[5] = shirt_class(image,detection[:4])
        tracks = tracker.update(Boxes(np.asarray(detections, dtype=np.float32).reshape(-1,6), (height,width)), image)
        players = []
        used = set()
        for tr in tracks:
            x1,y1,x2,y2,tid,score,cls,source_idx = tr
            tid = int(tid)
            # Break a recovered edge-exit identity rather than imply cross-exit ReID.
            old = previous_boxes.get(tid)
            if old and index-old[0] > 1 and old[1]:
                identities.mapping.pop(tid, None)
            # Match metadata geometrically; tracker indices can refer to filtered detections.
            boxes = np.asarray(detections)[:,:4]
            inter = np.maximum(0,np.minimum(boxes[:,2:],tr[2:4])-np.maximum(boxes[:,:2],tr[:2])).prod(axis=1)
            union = (boxes[:,2:]-boxes[:,:2]).prod(axis=1)+(x2-x1)*(y2-y1)-inter
            source_idx = int(np.argmax(inter/np.maximum(union,1)))
            team, role, source_id = labels[source_idx]
            edge = x1 < 3 or y1 < 3 or x2 > width-3 or y2 > height-3
            anchor = (source_id,int(cls)) if source_id is not None else None
            local_id = identities.link(tid,anchor,tuple(float(v) for v in tr[:4]),index,fps,used,edge)
            used.add(local_id)
            votes[local_id][(team,role)] += 1
            team,role = votes[local_id].most_common(1)[0][0]
            previous_boxes[tid] = (index, x1 < 3 or y1 < 3 or x2 > width-3 or y2 > height-3)
            box = [round(float(x1/width),5),round(float(y1/height),5),
                   round(float((x2-x1)/width),5),round(float((y2-y1)/height),5)]
            players.append({"id":local_id,"box":box,"team":team,"role":role,"shirt_class":int(cls),"predicted":False})
        frames.append({"frame":index,"scene":scene,"players":players,"balls":balls})
        if writer:
            for p in players:
                x,y,w,h = p["box"]
                x,y,w,h = int(x*width),int(y*height),int(w*width),int(h*height)
                color = (255,135,55) if p["shirt_class"] == 1 else (245,245,245)
                if p["role"] == "referee": color=(210,60,210)
                cv2.rectangle(image,(x,y),(x+w,y+h),color,1)
                label=f'#{p["id"]}'
                cv2.rectangle(image,(x,max(0,y-19)),(min(width,x+len(label)*10+5),max(19,y)),(20,24,28),-1)
                cv2.putText(image,label,(x+2,max(15,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
            for x,y,w,h in balls:
                cv2.rectangle(image,(int(x*width),int(y*height)),(int((x+w)*width),int((y+h)*height)),(0,220,255),1)
            writer.write(image)
        if index % 300 == 0:
            print(f"frame={index}/{total} scene={scene} local_ids={identities.next_id-1}",flush=True)
    cap.release()
    if writer: writer.release()
    data = {"version":1,"video":str(args.video.resolve()),"gsr_json":str(args.gsr_json.resolve()),
        "fps":fps,"width":width,"height":height,"duration":len(frames)/fps,"frames":frames,"cuts":cuts,
        "method":"BoT-SORT + jersey conflict gate + sparse optical-flow GMC + YOLO embeddings + gated GSR continuity",
        "identity_scope":"shot-local temporary IDs; no guaranteed identity through occlusion",
        "reviews":[]}
    (args.output_dir/"player_tracks.json").write_text(json.dumps(data,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    if writer:
        ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).with_name("ffmpeg"))
        subprocess.run([ffmpeg,"-y","-loglevel","error","-i",str(temporary),"-i",str(args.video),
            "-map","0:v","-map","1:a?","-c:v","libx264","-preset","fast","-crf","23",
            "-c:a","aac","-pix_fmt","yuv420p","-t",str(data["duration"]),"-movflags","+faststart",
            str(args.output_dir/"player_numbered.mp4")],check=True)
        temporary.unlink()
    print(f"Saved {len(frames)} frames to {args.output_dir}",flush=True)


if __name__ == "__main__":
    main()
