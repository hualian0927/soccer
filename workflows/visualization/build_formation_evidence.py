"""Generate optional frame/clip formation visuals and enrich an existing layered report."""

import argparse
from bisect import bisect_right
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from workflows.visualization.make_tactical_report_video import font
from workflows.identity.jersey_color import estimate_jersey_color
from tactical_analysis.gsr_io import load_gsr_context, infer_video_fps
from tactical_analysis.formation_evidence import describe_window, frame_distribution, scene_at
from tactical_analysis.reviewed_layers import organization_candidates, reviewed_statistics


COLORS = [(83,195,255),(102,224,161),(235,194,99),(226,147,225)]


def kit_consistency(image, distribution):
    """Reject strong within-team color conflicts; never reassign a team from one crop."""
    labels = []
    for line in distribution["lines"]:
        for player in line:
            b = player["box"]
            x,y,w,h = [round(b[k]) for k in ("x","y","w","h")]
            crop = image[max(0,y):min(image.shape[0],y+h),max(0,x):min(image.shape[1],x+w)]
            color = estimate_jersey_color(crop)
            if color.confidence >= .8 and color.scores.get(color.label,0) >= .25:
                labels.append(color.label)
    counts = {label:labels.count(label) for label in set(labels)}
    strong = sorted((count for label,count in counts.items() if label not in {"grey","black"}),reverse=True)
    return {"passed":not (len(strong)>1 and strong[1]>=2),"colorCounts":counts,
            "scope":"单帧球衣一致性筛查，不替代球队身份验证"}


def draw_distribution(image, distribution, label, time):
    if distribution:
        for index,line in enumerate(distribution["lines"]):
            color = COLORS[index % len(COLORS)]
            points = []
            for player in line:
                b = player["box"]
                x,y,w,h = [int(round(b[k])) for k in ("x","y","w","h")]
                cv2.rectangle(image,(x,y),(x+w,y+h),color[::-1],2)
                points.append((x+w//2,y+h))
            if len(points)>1:
                cv2.polylines(image,[np.array(points,dtype=np.int32)],False,color[::-1],2,cv2.LINE_AA)
    pil = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(pil,"RGBA")
    draw.rectangle((0,pil.height-76,pil.width,pil.height),fill=(15,20,23,235))
    draw.text((18,pil.height-67),f"{label}  |  原片 {time:.2f} 秒",font=font(21),fill=(245,245,245))
    text = "各色对应一条可见站位线；仅作投影候选，不补画未出镜球员。" if distribution else "当前帧人数或坐标不足，暂停绘制连线。"
    draw.text((18,pil.height-33),text,font=font(17),fill=(190,205,214))
    return cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)


def build(video, gsr_json, bundle_path, output_dir, tracking_path=None, review_path=None):
    output_dir.mkdir(parents=True,exist_ok=True)
    bundle = deepcopy(json.loads(bundle_path.read_text()))
    reviews = json.loads(review_path.read_text()) if review_path else {}
    if reviews and Path(reviews["sourceVideo"]).resolve() != video.resolve():
        raise ValueError("Formation review source does not match video")
    fps = infer_video_fps(video)
    context = load_gsr_context(gsr_json,fps,video)
    frame_map = {f.frame-1:f for f in context.frames}
    cap = cv2.VideoCapture(str(video))
    width,height = int(cap.get(3)),int(cap.get(4))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    if tracking_path:
        cuts = json.loads(tracking_path.read_text())["cuts"]
    else:
        # Conservative sequential cut scan; abrupt fades can split additional windows.
        cuts,previous,index = [],None,0
        while True:
            ok,image = cap.read()
            if not ok:break
            small=cv2.cvtColor(cv2.resize(image,(160,90)),cv2.COLOR_BGR2GRAY)
            if previous is not None and np.abs(small.astype(float)-previous).mean()>28 and cv2.matchTemplate(small,previous,cv2.TM_CCOEFF_NORMED)[0,0]<.55:
                cuts.append(index/fps)
            previous=small;index+=1
    sampled = [f for f in context.frames if (f.frame-1)%max(1,round(fps/2)) == 0]
    requests=[]
    for item in bundle["layers"].get("L4",{}).get("items",[]):
        if item["id"].startswith("formation"):
            team = "left" if item["id"].endswith("left") else "right"
            requests.append((item["time"],team))
    ranked=[]
    for f in sampled:
        for team in ("left","right"):
            estimate = frame_distribution(f,team)
            if estimate and estimate["line_separation_m"]>=5:
                ranked.append((estimate["visible_players"],estimate["confidence"],f.time_sec,team))
    for _,_,time,team in sorted(ranked,reverse=True):
        if len(requests)>=6:break
        if all(abs(time-old)>15 for old,_ in requests):requests.append((time,team))
    items=[]
    rejected=[]
    ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).with_name("ffmpeg"))
    for index,(time,team) in enumerate(requests):
        evidence=describe_window(sampled,team,time,cuts)
        if not evidence:continue
        representative=evidence["representative"]
        time=representative["time_sec"]
        scene=scene_at(time,cuts)
        review = next((r for r in reviews.get("frames",[]) if r["team"]==team and abs(r["time"]-time)<.01),None)
        if review and review["decision"]=="rejected":
            rejected.append({"time":time,"team":team,"reason":review["reason"],"source":"visual_frame_review"})
            continue
        start=max(0,time-2,([0]+cuts)[scene]);end=min(duration,time+2,(cuts+[duration])[scene])
        directory=output_dir/f"formation_{index:02d}"
        directory.mkdir(exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES,representative["frame"]-1)
        ok,image=cap.read()
        if not ok:continue
        identity_quality=kit_consistency(image,representative)
        if not identity_quality["passed"]:
            rejected.append({"time":time,"team":team,"reason":"同队球衣颜色冲突","quality":identity_quality})
            continue
        cv2.imwrite(str(directory/"original.jpg"),image)
        title=f"{'队伍A' if team=='left' else '队伍B'} · {evidence['title']}"
        cv2.imwrite(str(directory/"distribution.jpg"),draw_distribution(image.copy(),representative,title,time))
        temporary=directory/"silent.mp4"
        writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
        first,last=round(start*fps),round(end*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES,first)
        for number in range(first,last):
            ok,image=cap.read()
            if not ok:break
            f=frame_map.get(number)
            distribution=frame_distribution(f,team) if f else None
            if distribution and not kit_consistency(image,distribution)["passed"]:
                distribution=None
            # Changing groups are not presented as a stable named formation.
            label=title if distribution and distribution["line_counts"]==representative["line_counts"] else "站位变化 · 当前帧局部分组"
            writer.write(draw_distribution(image,distribution,label,number/fps))
        writer.release()
        subprocess.run([ffmpeg,"-y","-loglevel","error","-i",str(temporary),"-ss",str(first/fps),
            "-t",str((last-first)/fps),"-i",str(video),"-map","0:v","-map","1:a?","-c:v","libx264",
            "-preset","fast","-crf","22","-pix_fmt","yuv420p","-c:a","aac","-t",str((last-first)/fps),
            "-movflags","+faststart",str(directory/"distribution.mp4")],check=True)
        temporary.unlink()
        paths={"imagePath":str((directory/"distribution.jpg").resolve()),"originalPath":str((directory/"original.jpg").resolve()),
               "clipPath":str((directory/"distribution.mp4").resolve())}
        artifact={**evidence,**paths,"identityQuality":identity_quality,"frameReview":review,"time":time,"start":first/fps,"end":last/fps,"team":team}
        (directory/"evidence.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
        summary=f"代表帧检测到{representative['visible_players']}名外场球员，分线人数为{' / '.join(map(str,representative['line_counts']))}；同镜头窗口{evidence['sampleCount']}个采样中有{evidence['supportingSamples']}个支持该分组。"
        items.append({"id":f"formation-visual-{index}","kind":"formation_visual","time":time,"end":end,"team":team,
            "title":title,"summary":summary,"status":evidence["status"],"decision":"uncertain","publishedForStatistics":False,
            "thumbnailPath":paths["imagePath"],"formationEvidence":artifact,"limitations":evidence["limitations"],
            "commentary":{"observation":summary,"interpretation":"连线呈现同一时刻的纵向分层与横向展开。人数不全或分组不稳定时，仅描述可见队形，不确认完整阵型。",
                          "advice":"结合前后片段观察各线是否协同移动、持球侧是否收缩，以及远侧人员是否出画。",
                          "limitation":"队伍A/B沿用上游身份标签；阵型名称为未经过教练确认的几何候选，无法证明整场固定阵型。"}})
        print(f"Generated {title} at {time:.2f}s",flush=True)
    cap.release()
    for event in bundle["events"]:
        event["sceneId"]=scene_at(event["time"],cuts)
    bundle["organizationRuleCandidates"]=organization_candidates(bundle["events"])
    bundle["layers"]["L2"]["reviewedMetrics"]=reviewed_statistics(bundle["events"])
    retained=[]
    for old in bundle["layers"]["L4"].get("items",[]):
        if old.get("kind")=="formation_visual":continue
        if old["id"].startswith("formation"):
            paired=min(items,key=lambda x:abs(x["time"]-old["time"]),default=None)
            if paired and abs(paired["time"]-old["time"])<3:
                paired["previousReview"]=old
                continue
            if any(abs(x["time"]-old["time"])<3 for x in rejected):
                continue
        retained.append(old)
    bundle["layers"]["L4"]["items"]=items+retained
    bundle["formationEvidenceMetadata"]={"sourceVideo":str(video.resolve()),"gsrJson":str(gsr_json.resolve()),
        "sceneCuts":cuts,"artifactCount":len(items),"rejectedFrames":rejected,"apiCalls":0,"method":"shot_local_observed_lines_v1"}
    (output_dir/"layered_analysis.json").write_text(json.dumps(bundle,ensure_ascii=False,indent=2))
    return bundle


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video",type=Path,required=True)
    p.add_argument("--gsr-json",type=Path,required=True)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--tracking-json",type=Path)
    p.add_argument("--review-json",type=Path)
    a=p.parse_args()
    build(a.video,a.gsr_json,a.bundle,a.output_dir,a.tracking_json,a.review_json)
