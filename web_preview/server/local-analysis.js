import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { exportPlayerReview } from "./player-focus.js";


const TEST_5MIN_SHA256 = "3ceafa207e1c57465ed793f19ac762a4eeb45353d4691a03c6c8a1f7151a9ce5";
const MAX_FRONTEND_EVENTS = 120;
const demoResults = {
  "soccernet-mu-lei-40m": {
    video: "soccer_input_dataset/outputs/SoccerNet_MU_LEI_40min_v1/03_tactical_red_blue_v2_web.mp4",
    report: "soccer_input_dataset/outputs/SoccerNet_MU_LEI_40min_v1/tactical_report_v2/tactical_analysis_report.json",
    clips: "soccer_input_dataset/outputs/SoccerNet_MU_LEI_40min_v1/set_piece_clips_v2/set_piece_clips.json",
  },
};

const eventPresentation = {
  ball_out_of_play_candidate: ["出界", "出界候选", "yellow", "L1-02"],
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


async function fileExists(filePath) {
  try {
    return (await stat(filePath)).isFile();
  } catch {
    return false;
  }
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


function fileUrl(filePath, repoRoot) {
  return `/api/media?path=${encodeURIComponent(path.relative(repoRoot, filePath))}`;
}


async function serveMedia(request, response, requestUrl, repoRoot) {
  const relativePath = requestUrl.searchParams.get("path");
  if (!relativePath) {
    sendJson(response, 400, { error: "缺少媒体文件路径。" });
    return;
  }

  const resolvedRoot = path.resolve(repoRoot);
  const filePath = path.resolve(resolvedRoot, relativePath);
  if (filePath !== resolvedRoot && !filePath.startsWith(`${resolvedRoot}${path.sep}`)) {
    sendJson(response, 403, { error: "不允许访问项目目录之外的文件。" });
    return;
  }

  let fileInfo;
  try {
    fileInfo = await stat(filePath);
  } catch {
    sendJson(response, 404, { error: "媒体文件不存在。" });
    return;
  }
  if (!fileInfo.isFile()) {
    sendJson(response, 404, { error: "媒体文件不存在。" });
    return;
  }

  const extension = path.extname(filePath).toLowerCase();
  const contentTypes = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".json": "application/json; charset=utf-8",
  };
  const range = request.headers.range;
  response.setHeader("Accept-Ranges", "bytes");
  response.setHeader("Content-Type", contentTypes[extension] || "application/octet-stream");

  if (!range) {
    response.statusCode = 200;
    response.setHeader("Content-Length", fileInfo.size);
    if (request.method === "HEAD") response.end();
    else createReadStream(filePath).pipe(response);
    return;
  }

  const match = /^bytes=(\d*)-(\d*)$/.exec(range);
  if (!match) {
    response.statusCode = 416;
    response.setHeader("Content-Range", `bytes */${fileInfo.size}`);
    response.end();
    return;
  }
  const start = match[1] ? Number(match[1]) : 0;
  const requestedEnd = match[2] ? Number(match[2]) : fileInfo.size - 1;
  const end = Math.min(requestedEnd, fileInfo.size - 1);
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || start > end || start >= fileInfo.size) {
    response.statusCode = 416;
    response.setHeader("Content-Range", `bytes */${fileInfo.size}`);
    response.end();
    return;
  }

  response.statusCode = 206;
  response.setHeader("Content-Range", `bytes ${start}-${end}/${fileInfo.size}`);
  response.setHeader("Content-Length", end - start + 1);
  if (request.method === "HEAD") response.end();
  else createReadStream(filePath, { start, end }).pipe(response);
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


async function loadResult(videoPath, reportPath, clipIndexPath, cached, repoRoot) {
  const report = JSON.parse(await readFile(reportPath, "utf8"));
  const videoInfo = await stat(videoPath);
  const events = reportToEvents(report);
  let setPieceClips = [];
  try {
    const clips = JSON.parse(await readFile(clipIndexPath, "utf8"));
    setPieceClips = clips.map((clip) => ({
      ...clip,
      clipUrl: fileUrl(clip.clip_path, repoRoot),
      typeLabel: setPieceLabels[clip.event_subtype] || setPieceLabels.unknown,
      startPosition: [clip.start_x, clip.start_y],
      endPosition: [clip.end_x, clip.end_y],
    }));
  } catch {
    setPieceClips = [];
  }
  return {
    videoUrl: fileUrl(videoPath, repoRoot),
    reportUrl: fileUrl(reportPath, repoRoot),
    events,
    l1Summary: summarizeL1(events),
    setPieces: setPieceClips,
    sizeBytes: videoInfo.size,
    cached,
    pipeline: "SoccerNetGSR + 球场映射 + 关键事件识别 + 定位球切片 + H.264转码",
  };
}


async function loadDemoResult(resultId, repoRoot) {
  if (resultId === "test5-formation-focus") {
    const result = await loadDemoResult("test5-player-focus",repoRoot);
    return {...result,...await loadLayeredResult(
      path.join(repoRoot,"soccer_input_dataset/outputs/test_5min_layered_v2/test_5min_boxes.mp4"),
      path.join(repoRoot,"soccer_input_dataset/outputs/test_5min_formation_focus/layered_analysis.json"),true,repoRoot)};
  }
  if (resultId === "test5-player-focus") {
    const result = await loadLayeredResult(
      path.join(repoRoot,"soccer_input_dataset/outputs/test_5min_layered_v2/test_5min_boxes.mp4"),
      path.join(repoRoot,"soccer_input_dataset/outputs/test_5min_layered_v4/layered_analysis.json"),true,repoRoot);
    const playerTrackingPath = "soccer_input_dataset/outputs/test_5min_player_focus/player_tracks.json";
    return {...result,playerTrackingPath,playerTrackingHash:await sha256(path.join(repoRoot,playerTrackingPath)),
      playerTrackingUrl:fileUrl(path.join(repoRoot,playerTrackingPath),repoRoot),
      playerReviewsUrl:fileUrl(path.join(repoRoot,"soccer_input_dataset/outputs/test_5min_player_focus/player_reviews.json"),repoRoot)};
  }
  if (resultId === "test5-relay-sol-v1") {
    return loadLayeredResult(
      path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_layered_v2/test_5min_boxes.mp4"),
      path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_relay_sol_v1/layered_analysis.json"), true, repoRoot);
  }
  if (resultId === "test5-relay-terra-v1") {
    return loadLayeredResult(
      path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_layered_v2/test_5min_boxes.mp4"),
      path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_relay_terra_v1/layered_analysis.json"), true, repoRoot);
  }
  if (resultId === "test5-layered-v4") {
    return loadLayeredResult(
      path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_layered_v2/test_5min_boxes.mp4"),
      path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_layered_v4/layered_analysis.json"), true, repoRoot);
  }
  if (resultId === "test5-openai-v1") {
    return loadLayeredResult(
      path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_layered_v2/test_5min_boxes.mp4"),
      path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_openai_v1/layered_analysis.json"), true, repoRoot);
  }
  if (resultId === "test5-assistant-v3") {
    const directory = path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_assistant_review_v3");
    return loadLayeredResult(path.join(directory, "test_5min_assistant_review.mp4"), path.join(directory, "layered_analysis.json"), true, repoRoot);
  }
  if (resultId === "test5-layered-v2") {
    const directory = path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_layered_v2");
    return loadLayeredResult(path.join(directory, "test_5min_boxes.mp4"), path.join(directory, "layered_analysis.json"), true, repoRoot);
  }
  if (resultId === "test5-dense-vision-v1") {
    const outputDirectory = path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_dense_vision_v1");
    const videoPath = path.join(outputDirectory, "test_5min_dense_vision.mp4");
    const reportPath = path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_corner_review_v2/report/tactical_analysis_report.json");
    const reviewIds = ["event-0013", "set-piece-001", "event-0104", "event-0155", "event-0249", "event-0317", "event-0326", "event-0345"];
    const reviewPaths = reviewIds.map((id) => path.join(outputDirectory, "reviews", `${id}.deepseek.json`));
    if (!(await Promise.all([videoPath, reportPath, ...reviewPaths].map(fileExists))).every(Boolean)) {
      throw new Error("五分钟多帧视觉复核结果不完整，请先生成完整视频与复核文件");
    }
    const [videoInfo, ...reviews] = await Promise.all([
      stat(videoPath),
      ...reviewPaths.map((filePath) => readFile(filePath, "utf8").then(JSON.parse)),
    ]);
    const events = reviews.map((review) => {
      const candidate = review.candidate;
      const presentation = eventPresentation[candidate.event_type] || ["其他", "候选事件", "blue", "L1-01"];
      const supported = review.fusion_status === "visual_supported_candidate";
      const rejected = review.fusion_status === "rejected_candidate";
      const uncertainSetPiece = candidate.event_type === "set_piece_delivery_candidate" && !supported;
      const subtype = uncertainSetPiece ? "unknown" : (candidate.event_subtype || candidate.metrics?.set_piece_type || "unknown");
      const verdict = supported ? "视觉支持候选" : rejected ? "视觉否定，待人工复核" : "证据不足，待人工复核";
      const time = Number(candidate.time_sec);
      return {
        id: `vision:${candidate.event_id}`,
        time,
        end: Math.max(time + 3, Number(review.evidence_window_sec?.[1] || time + 4)),
        category: presentation[0],
        group: presentation[3],
        type: uncertainSetPiece ? "定位球待复核" : presentation[1],
        subtype,
        title: uncertainSetPiece ? "定位球类型待复核" : `${candidate.label_zh || presentation[1]}${rejected ? "（视觉否定）" : ""}`,
        summary: `${verdict}：${review.vision_review?.short_reason_zh || "需要回看原片"}`,
        detail: `复核 ${review.frame_count} 帧 · ${verdict}`,
        thumbnailUrl: fileUrl(path.join(outputDirectory, `${candidate.event_id}_poster.jpg`), repoRoot),
        confidence: Math.round(Number(candidate.confidence || 0) * 100),
        color: presentation[2],
        reviewStatus: "candidate",
        rawEventType: candidate.event_type,
      };
    }).sort((left, right) => left.time - right.time);
    return {
      videoUrl: fileUrl(videoPath, repoRoot),
      reportUrl: fileUrl(reportPath, repoRoot),
      events,
      l1Summary: summarizeL1(events),
      setPieces: [],
      sizeBytes: videoInfo.size,
      cached: true,
      pipeline: "原始五分钟视频 + 二维候选 + 8节点多帧视觉复核 + H.264完整视频；其余候选未复核",
    };
  }
  if (resultId === "full-pitch-ball-60s") {
    const videoPath = path.join(repoRoot, "二维分析/分析视频2.mp4");
    const reportPath = path.join(repoRoot, "二维分析/分析视频2_技战术结果/tactical_analysis_report.json");
    if (!(await fileExists(videoPath)) || !(await fileExists(reportPath))) {
      throw new Error("双机位足球分析结果不完整，请先运行 workflows/projection/analyze_projection_with_ball.py");
    }
    const [report, videoInfo] = await Promise.all([
      readFile(reportPath, "utf8").then(JSON.parse),
      stat(videoPath),
    ]);
    return {
      videoUrl: fileUrl(videoPath, repoRoot),
      reportUrl: fileUrl(reportPath, repoRoot),
      analysisMode: "spatial",
      spatialAnalysis: report,
      events: [],
      setPieces: [],
      sizeBytes: videoInfo.size,
      cached: true,
      pipeline: "双机位原画面 + 球员/足球投影 + 出界证据门控 + 分段空间指标",
    };
  }
  if (resultId === "full-pitch-spatial-60s") {
    const videoPath = path.join(repoRoot, "二维分析/分析视频.mp4");
    const reportPath = path.join(repoRoot, "二维分析/spatial_analysis_report.json");
    if (!(await fileExists(videoPath)) || !(await fileExists(reportPath))) {
      throw new Error("二维空间分析结果不完整，请先运行 workflows/projection/analyze_full_pitch_projection.py");
    }
    const [report, videoInfo] = await Promise.all([
      readFile(reportPath, "utf8").then(JSON.parse),
      stat(videoPath),
    ]);
    return {
      videoUrl: fileUrl(videoPath, repoRoot),
      reportUrl: fileUrl(reportPath, repoRoot),
      analysisMode: "spatial",
      spatialAnalysis: report,
      events: [],
      setPieces: [],
      sizeBytes: videoInfo.size,
      cached: true,
      pipeline: "双机位坐标融合 + 红蓝队去重 + 5秒空间指标 + 多帧视觉解释",
    };
  }
  const definition = demoResults[resultId];
  if (!definition) return null;
  const paths = Object.fromEntries(
    Object.entries(definition).map(([key, relativePath]) => [key, path.resolve(repoRoot, relativePath)]),
  );
  const available = await Promise.all(Object.values(paths).map(fileExists));
  if (!available.every(Boolean)) {
    throw new Error(`本地样例结果不完整：${resultId}`);
  }
  return loadResult(paths.video, paths.report, paths.clips, true, repoRoot);
}


async function processJob(job, { repoRoot }) {
  const useRelay = Boolean(process.env.RELAY_API_KEY && process.env.RELAY_BASE_URL);
  const useOpenAI = useRelay || Boolean(process.env.OPENAI_API_KEY);
  const reviewModel = useRelay ? (process.env.RELAY_MODEL || "gpt-5.6-terra") : "gpt-5.6-sol";
  const relayBase = useRelay ? process.env.RELAY_BASE_URL.replace(/\/+$/, "") : null;
  const reviewEndpoint = useRelay ? `${relayBase}${relayBase.endsWith("/v1") ? "" : "/v1"}/responses` : "https://api.openai.com/v1/responses";
  const knownDirectory = path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_layered_v2");
  const knownVideo = path.join(knownDirectory, "test_5min_boxes.mp4");
  const knownReport = useRelay
    ? path.join(repoRoot, `soccer_input_dataset/outputs/${reviewModel === "gpt-5.6-sol" ? "test_5min_relay_sol_v1" : "test_5min_relay_terra_v1"}/layered_analysis.json`) : useOpenAI
    ? path.join(repoRoot, "soccer_input_dataset/outputs/test_5min_openai_v1/layered_analysis.json")
    : await fileExists(path.join(repoRoot,"soccer_input_dataset/outputs/test_5min_formation_focus/layered_analysis.json"))
      ? path.join(repoRoot,"soccer_input_dataset/outputs/test_5min_formation_focus/layered_analysis.json")
      : path.join(knownDirectory, "layered_analysis.json");
  const logPath = path.join(job.directory, "analysis.log");
  try {
    job.stage = "正在校验素材与已有分析缓存";
    job.progress = 8;
    const digest = await sha256(job.inputPath);
    const cacheAvailable = await Promise.all([
      fileExists(knownVideo),
      fileExists(knownReport),
    ]).then((checks) => checks.every(Boolean));
    const knownReview = cacheAvailable && useOpenAI ? JSON.parse(await readFile(knownReport, "utf8")) : null;
    const reviewCacheValid = !useOpenAI || (knownReview?.reviewMetadata?.mode === (useRelay ? "compatible_api" : "openai_api")
      && knownReview.reviewMetadata.model === reviewModel
      && (!useRelay || knownReview.reviewMetadata.reasoning_effort === (process.env.RELAY_REASONING_EFFORT || "medium"))
      && (!useRelay || knownReview.reviewMetadata.response_format === (process.env.RELAY_RESPONSE_FORMAT || "schema"))
      && (knownReview.reviewMetadata.endpoint || "https://api.openai.com/v1/responses") === reviewEndpoint
      && knownReview.reviewMetadata.workflow_version === "2026-09-30-openai-v1"
      && knownReview.reviewMetadata.video_sha256 === digest
      && knownReview.reviewMetadata.boundary_transition_audit === true
      && !knownReview.reviewMetadata.task_states?.failed && !knownReview.reviewMetadata.task_states?.budget_exhausted);
    if (digest === TEST_5MIN_SHA256 && cacheAvailable && reviewCacheValid) {
      job.stage = "已命中 test_5min 完整分析结果";
      job.progress = 88;
      job.result = await loadLayeredResult(knownVideo, knownReport, true, repoRoot);
      const trackingPath = "soccer_input_dataset/outputs/test_5min_player_focus/player_tracks.json";
      if (await fileExists(path.join(repoRoot,trackingPath))) {
        Object.assign(job.result, {playerTrackingPath:trackingPath,playerTrackingHash:await sha256(path.join(repoRoot,trackingPath)),
          playerTrackingUrl:fileUrl(path.join(repoRoot,trackingPath),repoRoot),
          playerReviewsUrl:fileUrl(path.join(repoRoot,"soccer_input_dataset/outputs/test_5min_player_focus/player_reviews.json"),repoRoot)});
      }
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
    const identityPath = path.join(job.directory, "identities.json");

    job.stage = "正在执行球员、足球检测跟踪与球场映射";
    job.progress = 15;
    await runProcess(job, "conda", [
      "run", "-n", "sports", "python", "-m", "workflows.gsr.run_local_video_gsr_visualization",
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

    job.stage = "正在复核球衣与门将轨迹身份";
    job.progress = 62;
    await runProcess(job, "conda", [
      "run", "-n", "sports", "python", "-m", "workflows.identity.refine_gsr_video_identities",
      "--video", job.inputPath, "--json-path", jsonPath, "--output-json", identityPath,
    ], repoRoot, logPath);

    job.stage = "正在生成与本地项目一致的技战术可视化";
    job.progress = 66;
    await runProcess(job, "conda", [
      "run", "-n", "sports", "python", "-m", "workflows.visualization.make_tactical_visualization_video",
      "--input-video", job.inputPath,
      "--json-path", identityPath,
      "--output-video", tacticalVideo,
      "--boxes-only",
      "--yolo-fallback",
      "--fallback-backend", "ultralytics",
      "--team0-label", "蓝队",
      "--team1-label", "白队",
    ], repoRoot, logPath);

    job.stage = "正在生成事件时间轴与定位球分类";
    job.progress = 80;
    await runProcess(job, "conda", [
      "run", "-n", "sports", "python", "-m", "workflows.tactical.run_tactical_analysis",
      "--json-path", identityPath,
      "--input-video", job.inputPath,
      "--output-dir", reportDirectory,
    ], repoRoot, logPath);

    job.stage = useOpenAI ? "正在进行时序视觉审核、补充证据与战术解读"
      : process.env.DEEPSEEK_API_KEY ? "正在用前后5秒图像审核关键事件并生成分层解读" : "正在准备前后5秒审核证据与分层统计";
    job.progress = 88;
    const layeredDirectory = path.join(job.directory, "layered_review");
    await runProcess(job, "conda", [
      "run", "-n", "sports", "python", "-m", useOpenAI ? "workflows.review.run_openai_tactical_review" : "workflows.review.build_layered_tactical_review",
      "--video", job.inputPath,
      "--report-json", path.join(reportDirectory, "tactical_analysis_report.json"),
      "--output-dir", layeredDirectory,
      ...(useRelay ? ["--base-url", relayBase, "--model", reviewModel, "--api-key-env", "RELAY_API_KEY",
          "--reasoning-effort", process.env.RELAY_REASONING_EFFORT || "medium",
          "--response-format", process.env.RELAY_RESPONSE_FORMAT || "schema", "--no-cost-limit", "--max-calls", "150"]
        : useOpenAI ? ["--max-usd", process.env.OPENAI_REVIEW_MAX_USD || "8", "--max-calls", "80"]
        : process.env.DEEPSEEK_API_KEY ? ["--review-api"] : []),
    ], repoRoot, logPath);

    job.stage = "正在转换网页播放格式";
    job.progress = 92;
    await runProcess(job, "conda", [
      "run", "-n", "sports", "ffmpeg", "-y", "-loglevel", "error",
      "-i", tacticalVideo, "-i", job.inputPath, "-map", "0:v", "-map", "1:a?",
      "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
      "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-shortest", webVideo,
    ], repoRoot, logPath);

    job.result = await loadLayeredResult(
      webVideo,
      path.join(layeredDirectory, "layered_analysis.json"),
      false,
      repoRoot,
    );
    job.stage = "正在生成可选球员编号轨迹";
    job.progress = 96;
    const playerDirectory = path.join(job.directory,"player_focus");
    await runProcess(job,"conda",["run","-n","sports","python","-m", "workflows.players.build_player_focus_video",
      "--video",job.inputPath,"--gsr-json",identityPath,"--output-dir",playerDirectory],repoRoot,logPath);
    const trackingFile = path.join(playerDirectory,"player_tracks.json");
    job.stage = "正在生成可选阵型分布证据";
    job.progress = 98;
    const formationDirectory = path.join(job.directory,"formation_evidence");
    await runProcess(job,"conda",["run","-n","sports","python","-m", "workflows.visualization.build_formation_evidence",
      "--video",job.inputPath,"--gsr-json",identityPath,"--bundle",path.join(layeredDirectory,"layered_analysis.json"),
      "--tracking-json",trackingFile,"--output-dir",formationDirectory],repoRoot,logPath);
    job.result = await loadLayeredResult(webVideo,path.join(formationDirectory,"layered_analysis.json"),false,repoRoot);
    Object.assign(job.result,{playerTrackingPath:path.relative(repoRoot,trackingFile),playerTrackingUrl:fileUrl(trackingFile,repoRoot)});
    job.status = "completed";
    job.stage = job.result.reviewWarnings?.length ? "分析输出已生成，部分节点审核未完成" : "完整技战术分析已完成";
    job.progress = 100;
  } catch (error) {
    job.status = "failed";
    job.stage = "分析失败";
    job.error = `${error.message}，处理日志：${logPath}`;
  } finally {
    job.child = null;
  }
}


async function loadLayeredResult(videoPath, reportPath, cached, repoRoot) {
  const [report, videoInfo] = await Promise.all([readFile(reportPath, "utf8").then(JSON.parse), stat(videoPath)]);
  const events = report.events.map((event) => ({
    ...event,
    thumbnailUrl: event.thumbnailPath ? fileUrl(event.thumbnailPath, repoRoot) : null,
    reviewClipUrl: event.clipPath ? fileUrl(event.clipPath, repoRoot) : null,
    evidenceFrames: (event.evidenceFrames || []).map((frame) => ({time: frame.time, url: fileUrl(frame.path, repoRoot)})),
    inspectedFrames: (event.inspectedFrames || []).map((frame) => ({time: frame.time, url: fileUrl(frame.path, repoRoot)})),
  }));
  const layers = Object.fromEntries(Object.entries(report.layers).map(([key, layer]) => [key, {
    ...layer,
    ...(layer.items ? { items: layer.items.map((item) => ({
      ...item,
      thumbnailUrl: item.thumbnailPath ? fileUrl(item.thumbnailPath, repoRoot) : null,
      ...(item.formationEvidence ? {formationEvidence:{...item.formationEvidence,
        imageUrl:fileUrl(item.formationEvidence.imagePath,repoRoot),
        originalUrl:fileUrl(item.formationEvidence.originalPath,repoRoot),
        clipUrl:fileUrl(item.formationEvidence.clipPath,repoRoot)}} : {}),
      reviewClipUrl: item.clipPath ? fileUrl(item.clipPath, repoRoot) : null,
      inspectedFrames: (item.inspectedFrames || []).map((frame) => ({time: frame.time, url: fileUrl(frame.path, repoRoot)})),
    })) } : {}),
  }]));
  return {
    videoUrl: fileUrl(videoPath, repoRoot), reportUrl: fileUrl(reportPath, repoRoot),
    events, layers, reviewSummary: report.reviewSummary, boxesOnly: true,
    l1Summary: summarizeL1(events.filter((event) => event.publishedForStatistics !== false)), sizeBytes: videoInfo.size, cached,
    setPieces: events.filter((event) => event.category === "定位球" && event.reviewClipUrl && event.publishedForStatistics !== false).map((event) => ({
      event_id: event.id, clipUrl: event.reviewClipUrl, event_subtype: event.subtype,
      typeLabel: setPieceLabels[event.subtype] || "类型待复核", label_zh: event.title,
      timestamp_sec: event.time, clip_duration_sec: event.end - event.evidenceStart,
      summary_zh: event.summary, outcome: "unknown",
    })),
    reviewWarnings: ["openai_api", "compatible_api"].includes(report.reviewMetadata?.mode)
      && (report.reviewMetadata.task_states.failed || report.reviewMetadata.task_states.budget_exhausted)
      ? ["部分节点因审核失败或预算上限未完成，保留待核实，不计入采纳统计。"] : [],
    pipeline: ["openai_api", "compatible_api"].includes(report.reviewMetadata?.mode)
      ? "自动时序视觉复核 + 动态补充证据 + 门将复审；非人工专家真值"
      : report.reviewMetadata ? "助手直接抽帧复核样例；非自动分析或人工专家真值" : "检测框原片 + 前后各5秒图像复核 + 关键事件/数据统计/进攻组织/队形空间",
  };
}


export function localAnalysisPlugin({ repoRoot }) {
  const jobs = new Map();
  const runtimeRoot = path.join(repoRoot, "web_preview/runtime");

  return {
    name: "local-football-analysis-api",
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const requestUrl = new URL(request.url, "http://localhost");
        if (request.method === "POST" && requestUrl.pathname === "/api/player-review") {
          try {sendJson(response,200,await exportPlayerReview(await readJsonBody(request),repoRoot));}
          catch(error) {sendJson(response,400,{error:error.message});}
          return;
        }
        if ((request.method === "GET" || request.method === "HEAD") && requestUrl.pathname === "/api/media") {
          await serveMedia(request, response, requestUrl, repoRoot);
          return;
        }
        const demoResultMatch = requestUrl.pathname.match(/^\/api\/demo-results\/([a-z0-9-]+)$/i);
        if (request.method === "GET" && demoResultMatch) {
          try {
            const result = await loadDemoResult(demoResultMatch[1], repoRoot);
            if (!result) sendJson(response, 404, { error: "找不到该本地样例结果。" });
            else sendJson(response, 200, result);
          } catch (error) {
            sendJson(response, 500, { error: error.message });
          }
          return;
        }
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
          else if (!job.result) sendJson(response, 409, { error: "关键事件分析尚未完成。", ...publicJob(job) });
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
            sendJson(response, 404, { error: "找不到可审核的事件分析结果。" });
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
              sendJson(response, 404, { error: "找不到该关键事件。" });
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
