"""Export a selected temporary player ID, evidence frames and a gated report. No API calls."""

import argparse
import hashlib
import json
import subprocess
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tactical_analysis.player_focus import summarize_selection


def export_selection(tracking, player_id, start, end, output):
    data = json.loads(tracking.read_text())
    tracking_digest = hashlib.sha256(tracking.read_bytes()).hexdigest()
    report = summarize_selection(data, player_id, start, end)
    output.mkdir(parents=True, exist_ok=True)
    reviews_path = tracking.parent / "player_reviews.json"
    reviews = json.loads(reviews_path.read_text()) if reviews_path.exists() else []
    report["tracking_sha256"] = tracking_digest
    report["review"] = next((r for r in reviews if r.get("tracking_sha256") == tracking_digest and r["player_id"] == player_id and
        abs(r["start"]-start) < .1 and abs(r["end"]-end) < .1), None)
    if report["review"]:
        report["status"] = "visual_reviewed_example"
    fps, width, height = data["fps"], data["width"], data["height"]
    cap = cv2.VideoCapture(data["video"])
    first, last = int(round(start*fps)), int(round(end*fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer = cv2.VideoWriter(str(output / "silent.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if not writer.isOpened():
        raise ValueError("Cannot open video writer")
    indices = set(np.linspace(first,max(first,last-1),12).astype(int).tolist())
    tiles = []
    for index in range(first,last):
        ok, image = cap.read()
        if not ok: break
        frame = data["frames"][index]
        target = next((p for p in frame["players"] if p["id"] == player_id),None)
        if target:
            x,y,w,h = target["box"]
            x,y,w,h = int(x*width),int(y*height),int(w*width),int(h*height)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,220,255),3)
            cv2.putText(image,f"#{player_id}",(max(0,x),max(25,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,220,255),2,cv2.LINE_AA)
        cv2.putText(image,f"Source {index/fps:.2f}s",(20,height-20),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
        writer.write(image)
        if index in indices:
            path = output / f"frame_{index:06d}.jpg"
            cv2.imwrite(str(path), image)
            tile = Image.fromarray(cv2.cvtColor(cv2.resize(image,(640,360)),cv2.COLOR_BGR2RGB))
            tiles.append(tile)
    cap.release()
    writer.release()
    sheet = Image.new("RGB",(1280,360*((len(tiles)+1)//2)),"black")
    for i,tile in enumerate(tiles): sheet.paste(tile,((i%2)*640,(i//2)*360))
    sheet.save(output / "evidence.jpg")
    ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).with_name("ffmpeg"))
    subprocess.run([ffmpeg,"-y","-loglevel","error","-i",str(output/"silent.mp4"),
        "-ss",str(start),"-t",str(end-start),"-i",data["video"],"-map","0:v","-map","1:a?",
        "-c:v","libx264","-preset","fast","-crf","21","-c:a","aac","-pix_fmt","yuv420p",
        "-t",str(end-start),"-movflags","+faststart",str(output/"player_clip.mp4")],check=True)
    (output/"silent.mp4").unlink()
    report["frames"] = [str(p.resolve()) for p in sorted(output.glob("frame_*.jpg"))]
    report["clip"] = str((output/"player_clip.mp4").resolve())
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    text = [f"# 球员 #{player_id} 片段分析",f"原片时间：{start:.2f}–{end:.2f} 秒",f"可见覆盖率：{report['coverage']:.1%}",
            f"脚边球框邻近帧数：{report['ball_near_feet_frames']}（不等于触球次数）",*report["limitations"]]
    if report["review"]:
        for key in ("observation","analysis","advice","limitations"):
            text.append(f"{key}：{report['review'].get(key,'')}")
    else:
        text.append("本片段尚未进行画面语义复核，不自动判断传球成功、射门、越位或动作规范性。")
    (output/"report.md").write_text("\n\n".join(text),encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracking-json",type=Path,required=True)
    p.add_argument("--player-id",type=int,required=True)
    p.add_argument("--start",type=float,required=True)
    p.add_argument("--end",type=float,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a = p.parse_args()
    export_selection(a.tracking_json,a.player_id,a.start,a.end,a.output_dir)
