import { spawn } from "node:child_process";
import { readFile, realpath, mkdir } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import path from "node:path";

let busy = false;
const media = (p, root) => `/api/media?path=${encodeURIComponent(path.relative(root, p))}`;

export async function exportPlayerReview(body, root) {
  if (busy) throw new Error("另一个个人片段正在导出，请稍后重试");
  busy = true;
  try {
  const id = Number(body.playerId), start = Number(body.start), end = Number(body.end);
  if (!Number.isSafeInteger(id) || id < 1 || ![start,end].every(Number.isFinite) || start < 0 || end <= start || end-start > 30) {
    throw new Error("请选择有效球员及不超过30秒的同镜头片段");
  }
  const tracking = await realpath(path.resolve(root, String(body.trackingPath)));
  await mkdir(path.join(root,"web_preview/runtime"),{recursive:true});
  await mkdir(path.join(root,"soccer_input_dataset/outputs"),{recursive:true});
  const roots = await Promise.all(["soccer_input_dataset/outputs", "web_preview/runtime"].map((p) => realpath(path.join(root,p))));
  if (path.basename(tracking) !== "player_tracks.json" || !roots.some((p) => tracking.startsWith(p + path.sep))) throw new Error("无效跟踪文件");
  const data = JSON.parse(await readFile(tracking,"utf8"));
  const video = await realpath(data.video);
  if (!video.startsWith(await realpath(root) + path.sep) || !/\.(mp4|mov|mkv|avi)$/i.test(video)) throw new Error("无效源视频");
  const output = path.join(root,"web_preview/runtime/player-reviews",randomUUID());
  await mkdir(output,{recursive:true});
    await new Promise((resolve,reject) => {
      const prefix = process.env.SPORTS_PYTHON ? [] : ["run","--no-capture-output","-n","sports","python"];
      const child = spawn(process.env.SPORTS_PYTHON || "conda",
        [...prefix,"-m", "workflows.players.export_player_focus","--tracking-json",tracking,"--player-id",String(id),
          "--start",String(start),"--end",String(end),"--output-dir",output], {cwd:root,stdio:["ignore","ignore","pipe"]});
      let error = "";
      child.stderr.on("data",(chunk) => {error = (error + chunk.toString()).slice(-1500);});
      const timer = setTimeout(() => child.kill("SIGTERM"),120000);
      child.on("error",(e) => {clearTimeout(timer);reject(e);});
      child.on("close",(code) => {clearTimeout(timer); code === 0 ? resolve() : reject(new Error(error.includes("camera cut") ? "所选片段跨越切镜，请缩短时间范围" : "片段导出失败，请确认球员在所选区间内可见"));});
    });
    const report = JSON.parse(await readFile(path.join(output,"report.json"),"utf8"));
    return {...report, clipUrl:media(report.clip,root), posterUrl:report.frames[0] ? media(report.frames[0],root) : null,
      evidenceUrl:media(path.join(output,"evidence.jpg"),root),
      reportUrl:media(path.join(output,"report.md"),root)};
  } finally {busy = false;}
}
