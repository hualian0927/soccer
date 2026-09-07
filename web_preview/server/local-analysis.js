import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { pipeline } from "node:stream/promises";


const TEST_5MIN_SHA256 = "3ceafa207e1c57465ed793f19ac762a4eeb45353d4691a03c6c8a1f7151a9ce5";
const MAX_FRONTEND_EVENTS = 120;

const eventPresentation = {
  shot_candidate: ["射门", "射门", "orange", "L1-01"],
  corner_candidate: ["定位球", "角球", "yellow", "L1-02"],
  set_piece_delivery_candidate: ["定位球", "定位球", "yellow", "L1-02"],
  touch_candidate: ["传接带", "稳定触球", "blue", "L1-03"],
  receive_candidate: ["传接带", "接球", "blue", "L1-03"],
  carry_candidate: ["传接带", "持球推进", "green", "L1-03"],
  pass_candidate: ["传接带", "传球", "blue", "L1-03"],
  possession_change: ["球权转换", "球权转换", "cyan", "L1-04"],
  defensive_intervention_candidate: ["防守干预", "防守干预", "red", "L1-05"],
  goalkeeper_intervention_candidate: ["门将事件", "门将干预", "violet", "L1-06"],
};

const setPieceLabels = {
  corner: "角球",
  free_kick: "任意球",
  goal_kick: "球门球",
  throw_in: "界外球",
  kickoff: "中圈开球",
  penalty: "点球",
  unknown: "未分类定位球",
};


function sendJson(response, statusCode, payload) {
  response.statusCode = statusCode;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.end(JSON.stringify(payload));
}


async function readJsonBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}


function sanitizeFilename(value) {
  const decoded = decodeURIComponent(value || "match.mp4");
  const basename = path.basename(decoded).replace(/[^a-zA-Z0-9._-]+/g, "_");
  return basename || "match.mp4";
}


async function sha256(filePath) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(filePath)) digest.update(chunk);
  return digest.digest("hex");
}


function metricSummary(metrics = {}) {
  const labels = {
    duration_sec: "持续",
    forward_progress_m: "向前推进",
    total_forward_progress_m: "总推进",
    transition_speed_mps: "转换速度",
    mean_advantage: "平均人数优势",
    maximum_line_gap_m: "最大线距",
    risk_score: "风险分",
    pass_count: "传球",
    width_m: "宽度",
    depth_m: "纵深",
    vote_ratio: "支持率",
    ball_displacement_m: "足球位移",
    flight_distance_m: "运行距离",
    flight_duration_sec: "运行时间",
    minimum_ball_distance_m: "球门将最近距离",
    approach_distance_drop_m: "接近距离变化",
    proximity_duration_sec: "接近持续",
  };
  const units = {
    duration_sec: " 秒",
    forward_progress_m: " 米",
    total_forward_progress_m: " 米",
    transition_speed_mps: " m/s",
    maximum_line_gap_m: " 米",
    width_m: " 米",
    depth_m: " 米",
    ball_displacement_m: " 米",
    flight_distance_m: " 米",
    flight_duration_sec: " 秒",
    minimum_ball_distance_m: " 米",
    approach_distance_drop_m: " 米",
    proximity_duration_sec: " 秒",
  };
  const items = Object.entries(labels)
    .filter(([key]) => Number.isFinite(metrics[key]))
    .slice(0, 3)
    .map(([key, label]) => {
      const value = key === "vote_ratio" ? `${Math.round(metrics[key] * 100)}%` : Number(metrics[key]).toFixed(1);
      return `${label} ${value}${units[key] || ""}`;
    });
  return items.join(" · ") || "来源于本地视觉技战术分析报告";
}


function reportToEvents(report) {
  const candidates = [];
  const reportDuration = Number(report.source?.duration_sec || Number.POSITIVE_INFINITY);
  for (const output of report.analyzer_outputs || []) {
    for (const event of output.events || []) {
      const presentation = eventPresentation[event.event_type];
      if (!presentation) continue;
      const metrics = event.metrics || {};
      const time = Number(event.time_sec || metrics.start_sec || 0);
      const subtype = event.event_subtype || metrics.set_piece_type || metrics.action_type || metrics.intervention_subtype || "unknown";
      const fallbackEnd = event.event_type.includes("set_piece") ? Number(metrics.landing_sec || time) + 4 : time + Math.max(3, Number(metrics.duration_sec || 0));
      const end = Math.min(reportDuration, Number(event.evidence_end_sec || metrics.end_sec || fallbackEnd));
      const title = event.event_type === "set_piece_delivery_candidate"
        ? `${setPieceLabels[subtype] || "定位球"}片段候选`
        : event.label_zh || presentation[1];
      candidates.push({
        id: `${output.analyzer}:${event.event_id}`,
        time,
        end,
        category: presentation[0],
        group: presentation[3],
        type: event.event_type === "set_piece_delivery_candidate" ? setPieceLabels[subtype] : presentation[1],
        subtype,
        title,
        summary: `${event.team ? `${event.team === "left" ? "左队" : "右队"}，` : ""}${metricSummary(metrics)}`,
        detail: metricSummary(metrics),
        confidence: Math.round(Number(event.confidence || 0.65) * 100),
        color: presentation[2],
        outcome: event.outcome || (metrics.retained_by_taking_team ? "retained" : "unknown"),
        team: event.team || null,
        actorTrackId: event.actor_track_id,
        recipientTrackId: event.target_track_id,
        startPosition: [event.start_x ?? metrics.origin_x ?? metrics.start_x, event.start_y ?? metrics.origin_y ?? metrics.start_y],
        endPosition: [event.end_x ?? metrics.landing_x ?? metrics.end_x, event.end_y ?? metrics.landing_y ?? metrics.end_y],
        attributes: metrics,
        reviewStatus: event.review_status || "candidate",
        rawEventType: event.event_type,
      });
    }
  }

  const fullSetPieces = candidates.filter((item) => item.category === "定位球" && item.id.startsWith("set_piece_delivery:"));
  const withoutDuplicateCorners = candidates.filter((item) => (
    !item.id.startsWith("event_timeline:")
    || item.subtype !== "corner"
    || !fullSetPieces.some((setPiece) => Math.abs(setPiece.time - item.time) < 3)
  ));
  const sampleEvenly = (items, limit) => {
    if (items.length <= limit) return items;
    if (limit <= 1) return items.slice(0, limit);
    return Array.from({ length: limit }, (_, index) => items[Math.round(index * (items.length - 1) / (limit - 1))]);
  };
  const quotas = {
    shot_candidate: 24,
    set_piece_delivery_candidate: 24,
    corner_candidate: 24,
    goalkeeper_intervention_candidate: 18,
    carry_candidate: 18,
    pass_candidate: 18,
    receive_candidate: 12,
    possession_change: 16,
    defensive_intervention_candidate: 16,
    touch_candidate: 8,
  };
  const selectedByType = Object.entries(quotas).flatMap(([type, limit]) => (
    sampleEvenly(withoutDuplicateCorners.filter((item) => item.rawEventType === type), limit)
  ));
  const ordered = selectedByType.sort((left, right) => left.time - right.time || right.confidence - left.confidence);
  const deduplicated = [];
  for (const event of ordered) {
    const existing = deduplicated.find((item) => Math.abs(item.time - event.time) < 0.15 && item.type === event.type);
    if (!existing) deduplicated.push(event);
    else if (event.confidence > existing.confidence) Object.assign(existing, event);
  }
  deduplicated.sort((left, right) => left.time - right.time);
  if (deduplicated.length <= MAX_FRONTEND_EVENTS) return deduplicated;
  const priority = {
    set_piece_delivery_candidate: 10,
    corner_candidate: 9,
    shot_candidate: 8,
    goalkeeper_intervention_candidate: 7,
    carry_candidate: 6,
    pass_candidate: 5,
    receive_candidate: 4,
    possession_change: 3,
    defensive_intervention_candidate: 2,
    touch_candidate: 1,
  };
  return deduplicated
    .sort((left, right) => (priority[right.rawEventType] || 0) - (priority[left.rawEventType] || 0) || left.time - right.time)
    .slice(0, MAX_FRONTEND_EVENTS)
    .sort((left, right) => left.time - right.time);
}


function summarizeL1(events) {
  const groupCounts = {};
  const setPieceTypeCounts = {};
  for (const event of events) {
    groupCounts[event.group] = (groupCounts[event.group] || 0) + 1;
    if (event.category === "定位球") {
      setPieceTypeCounts[event.subtype] = (setPieceTypeCounts[event.subtype] || 0) + 1;
    }
  }
  return {
    totalEvents: events.length,
    groupCounts,
    setPieceTypeCounts,
    setPieceTotal: events.filter((item) => item.category === "定位球").length,
  };
}


function fileUrl(filePath) {
  return `/@fs${filePath}`;
}


function publicJob(job) {
  return {
    id: job.id,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    error: job.error || null,
    result: job.result || null,
  };
}


function runProcess(job, executable, args, cwd, logPath) {
  return new Promise((resolve, reject) => {
    const output = createWriteStream(logPath, { flags: "a" });
    output.write(`\n$ ${executable} ${args.join(" ")}\n`);
    const child = spawn(executable, args, { cwd, env: process.env });
    child.stdout.pipe(output, { end: false });
    child.stderr.pipe(output, { end: false });
    child.on("error", reject);
    child.on("exit", (code) => {
      output.end();
      if (code === 0) resolve();
      else reject(new Error(`处理命令退出，状态码 ${code}`));
    });
    job.child = child;
  });
}


async function loadResult(videoPath, reportPath, clipIndexPath, cached) {
  const report = JSON.parse(await readFile(reportPath, "utf8"));
  const videoInfo = await stat(videoPath);
  const events = reportToEvents(report);
  let setPieceClips = [];
  try {
    const clips = JSON.parse(await readFile(clipIndexPath, "utf8"));
    setPieceClips = clips.map((clip) => ({
      ...clip,
      clipUrl: fileUrl(clip.clip_path),
      typeLabel: setPieceLabels[clip.event_subtype] || setPieceLabels.unknown,
      startPosition: [clip.start_x, clip.start_y],
      endPosition: [clip.end_x, clip.end_y],
    }));
  } catch {
    setPieceClips = [];
  }
  return {
    videoUrl: fileUrl(videoPath),
    reportUrl: fileUrl(reportPath),
    events,
    l1Summary: summarizeL1(events),
    setPieces: setPieceClips,
    sizeBytes: videoInfo.size,
    cached,
    pipeline: "SoccerNetGSR + 球场映射 + L1事件识别 + 定位球切片 + H.264转码",
  };
}


async function processJob(job, { repoRoot }) {
  const knownVideo = path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_current/test_5min_tactical_base_web.mp4");
  const knownReportDirectory = path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_current/l1_report_v1");
  const knownReport = path.join(knownReportDirectory, "tactical_analysis_report.json");
  const logPath = path.join(job.directory, "analysis.log");
  try {
    job.stage = "正在校验素材与已有分析缓存";
    job.progress = 8;
    const digest = await sha256(job.inputPath);
    if (digest === TEST_5MIN_SHA256) {
      job.stage = "已命中 test_5min 完整分析结果";
      job.progress = 88;
      const clipDirectory = path.join(job.directory, "set_piece_clips");
      await runProcess(job, "conda", [
        "run", "-n", "sports", "python", path.join(repoRoot, "export_set_piece_clips.py"),
        "--input-video", knownVideo,
        "--manifest", path.join(knownReportDirectory, "set_piece_manifest.csv"),
        "--output-dir", clipDirectory,
      ], repoRoot, logPath);
      job.result = await loadResult(knownVideo, knownReport, path.join(clipDirectory, "set_piece_clips.json"), true);
      job.status = "completed";
      job.progress = 100;
      return;
    }

    const videoName = `SNGS-${Date.now().toString().slice(-8)}`;
    const workRoot = path.join(job.directory, "gsr_work");
    const gsrPreview = path.join(job.directory, "gsr_preview.mp4");
    const tacticalVideo = path.join(job.directory, "tactical_base.mp4");
    const webVideo = path.join(job.directory, "tactical_base_web.mp4");
    const reportDirectory = path.join(job.directory, "tactical_report");
    const jsonPath = path.join(workRoot, "SoccerNetGS/test", videoName, `${videoName}.json`);

    job.stage = "正在执行球员、足球检测跟踪与球场映射";
    job.progress = 15;
    await runProcess(job, "conda", [
      "run", "-n", "sports", "python", path.join(repoRoot, "run_local_video_gsr_visualization.py"),
      "--input-video", job.inputPath,
      "--video-name", videoName,
      "--max-frames", "0",
      "--output-video", gsrPreview,
      "--work-root", workRoot,
      "--batch-size", "4",
      "--recall-optimized",
      "--overwrite",
      "--team0-colors", "blue",
      "--team1-colors", "white",
      "--referee-colors", "red",
      "--goalkeeper-team0-colors", "yellowgreen",
    ], repoRoot, logPath);

    job.stage = "正在生成与本地项目一致的技战术可视化";
    job.progress = 66;
    await runProcess(job, "conda", [
      "run", "-n", "sports", "python", path.join(repoRoot, "make_tactical_visualization_video.py"),
      "--input-video", job.inputPath,
      "--json-path", jsonPath,
      "--output-video", tacticalVideo,
      "--yolo-fallback",
      "--fallback-backend", "ultralytics",
      "--team0-label", "蓝队",
      "--team1-label", "白队",
    ], repoRoot, logPath);

    job.stage = "正在生成 L1 事件时间轴与定位球分类";
    job.progress = 80;
    await runProcess(job, "conda", [
      "run", "-n", "sports", "python", path.join(repoRoot, "run_tactical_analysis.py"),
      "--json-path", jsonPath,
      "--input-video", job.inputPath,
      "--output-dir", reportDirectory,
    ], repoRoot, logPath);

    job.stage = "正在按定位球事件切出独立片段";
    job.progress = 88;
    const clipDirectory = path.join(job.directory, "set_piece_clips");
    await runProcess(job, "conda", [
      "run", "-n", "sports", "python", path.join(repoRoot, "export_set_piece_clips.py"),
      "--input-video", tacticalVideo,
      "--manifest", path.join(reportDirectory, "set_piece_manifest.csv"),
      "--output-dir", clipDirectory,
    ], repoRoot, logPath);

    job.stage = "正在转换网页播放格式";
    job.progress = 92;
    await runProcess(job, "conda", [
      "run", "-n", "sports", "ffmpeg", "-y", "-loglevel", "error",
      "-i", tacticalVideo,
      "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
      "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", webVideo,
    ], repoRoot, logPath);

    job.result = await loadResult(
      webVideo,
      path.join(reportDirectory, "tactical_analysis_report.json"),
      path.join(clipDirectory, "set_piece_clips.json"),
      false,
    );
    job.status = "completed";
    job.stage = "完整技战术分析已完成";
    job.progress = 100;
  } catch (error) {
    job.status = "failed";
    job.stage = "分析失败";
    job.error = `${error.message}，处理日志：${logPath}`;
  } finally {
    job.child = null;
  }
}


export function localAnalysisPlugin({ repoRoot }) {
  const jobs = new Map();
  const runtimeRoot = path.join(repoRoot, "web_preview/runtime");

  return {
    name: "local-football-analysis-api",
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const requestUrl = new URL(request.url, "http://localhost");
        if (request.method === "POST" && requestUrl.pathname === "/api/analysis") {
          const id = randomUUID();
          const directory = path.join(runtimeRoot, "jobs", id);
          await mkdir(directory, { recursive: true });
          const filename = sanitizeFilename(request.headers["x-file-name"]);
          const inputPath = path.join(directory, filename);
          const job = {
            id,
            directory,
            inputPath,
            filename,
            status: "uploading",
            stage: "正在保存上传视频",
            progress: 2,
          };
          jobs.set(id, job);
          try {
            await pipeline(request, createWriteStream(inputPath));
            job.status = "processing";
            processJob(job, { repoRoot });
            sendJson(response, 202, publicJob(job));
          } catch (error) {
            job.status = "failed";
            job.error = error.message;
            sendJson(response, 500, publicJob(job));
          }
          return;
        }

        const match = requestUrl.pathname.match(/^\/api\/analysis\/([a-f0-9-]+)$/i);
        if (request.method === "GET" && match) {
          const job = jobs.get(match[1]);
          if (!job) sendJson(response, 404, { error: "找不到该分析任务，开发服务器可能已经重启。" });
          else sendJson(response, 200, publicJob(job));
          return;
        }
        const l1Match = requestUrl.pathname.match(/^\/api\/analysis\/([a-f0-9-]+)\/l1$/i);
        if (request.method === "GET" && l1Match) {
          const job = jobs.get(l1Match[1]);
          if (!job) sendJson(response, 404, { error: "找不到该分析任务。" });
          else if (!job.result) sendJson(response, 409, { error: "L1 分析尚未完成。", ...publicJob(job) });
          else sendJson(response, 200, {
            summary: job.result.l1Summary,
            events: job.result.events,
            setPieces: job.result.setPieces,
          });
          return;
        }
        const reviewMatch = requestUrl.pathname.match(/^\/api\/analysis\/([a-f0-9-]+)\/events\/([^/]+)$/i);
        if (request.method === "PATCH" && reviewMatch) {
          const job = jobs.get(reviewMatch[1]);
          if (!job?.result) {
            sendJson(response, 404, { error: "找不到可审核的 L1 分析结果。" });
            return;
          }
          try {
            const eventId = decodeURIComponent(reviewMatch[2]);
            const payload = await readJsonBody(request);
            const allowed = new Set(["candidate", "confirmed", "rejected"]);
            if (!allowed.has(payload.reviewStatus)) {
              sendJson(response, 400, { error: "reviewStatus 必须是 candidate、confirmed 或 rejected。" });
              return;
            }
            const event = job.result.events.find((item) => item.id === eventId);
            if (!event) {
              sendJson(response, 404, { error: "找不到该 L1 事件。" });
              return;
            }
            event.reviewStatus = payload.reviewStatus;
            const reviews = job.reviews || [];
            const record = {
              eventId,
              reviewStatus: payload.reviewStatus,
              updatedAt: new Date().toISOString(),
            };
            const oldIndex = reviews.findIndex((item) => item.eventId === eventId);
            if (oldIndex >= 0) reviews[oldIndex] = record;
            else reviews.push(record);
            job.reviews = reviews;
            await writeFile(path.join(job.directory, "review_log.json"), JSON.stringify(reviews, null, 2), "utf8");
            sendJson(response, 200, { event, review: record });
          } catch (error) {
            sendJson(response, 400, { error: error.message });
          }
          return;
        }
        next();
      });
    },
  };
}
