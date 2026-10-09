import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";


const dirname = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(dirname, "..");
const repoRoot = path.resolve(appRoot, "..");
const output = path.join(appRoot, "test-results");
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_PATH
  || path.join(process.env.HOME, ".cache/ms-playwright/chromium-1234/chrome-linux64/chrome");
const localLibraries = path.join(appRoot, "test-runtime/root/usr/lib/x86_64-linux-gnu");
const url = "http://127.0.0.1:4173/";

await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  executablePath,
  env: {
    ...process.env,
    LD_LIBRARY_PATH: `${localLibraries}:${process.env.LD_LIBRARY_PATH || ""}`,
  },
});

async function assertNoHorizontalOverflow(page, label) {
  const dimensions = await page.evaluate(() => ({
    viewport: window.innerWidth,
    document: document.documentElement.scrollWidth,
  }));
  if (dimensions.document > dimensions.viewport + 1) {
    throw new Error(`${label} horizontal overflow: ${JSON.stringify(dimensions)}`);
  }
}

const desktop = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
const page = await desktop.newPage();
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") errors.push(message.text());
});

await page.goto(url, { waitUntil: "networkidle" });
await page.getByRole("heading", { name: "比赛素材库" }).waitFor();
await page.waitForFunction(() => {
  const posters = [...document.querySelectorAll(".asset-poster img")];
  return posters.length === document.querySelectorAll(".asset-card").length
    && posters.length >= 6
    && posters.every((poster) => poster.complete && poster.naturalWidth > 0);
});
await page.locator(".asset-card").filter({ has: page.getByRole("heading", { name: "SoccerNet 40分钟长视频测试", exact: true }) }).waitFor();
await page.screenshot({ path: path.join(output, "desktop-assets.png"), fullPage: true });

const correctedAsset = page.locator(".asset-card").filter({ has: page.getByRole("heading", { name: "五分钟定位球候选复核（修正版）", exact: true }) });
await correctedAsset.getByRole("button", { name: "预览" }).click();
await page.getByRole("dialog").getByText("定位球结论仍需人工复核").waitFor();
await page.waitForFunction(() => {
  const video = document.querySelector(".preview-dialog video");
  return video && video.readyState >= 1 && video.duration >= 299;
});
await page.screenshot({ path: path.join(output, "desktop-corrected-review.png") });
await page.locator(".preview-dialog .icon-button").click();

const visionPage = await desktop.newPage();
visionPage.on("pageerror", (error) => errors.push(`five-minute vision: ${error.message}`));
await visionPage.goto(url, { waitUntil: "networkidle" });
await visionPage.getByRole("button", { name: "项目", exact: true }).click();
await visionPage.getByRole("button", { name: /五分钟多帧视觉复核/ }).click();
await visionPage.locator(".project-video-row").filter({ hasText: "五分钟多帧视觉复核测试" }).click();
await visionPage.getByRole("heading", { name: "关键事件" }).waitFor();
if (await visionPage.locator(".segment-row").count() !== 8) {
  throw new Error("five-minute vision project did not load all 8 reviewed nodes");
}
const visionThumbnails = await visionPage.locator(".segment-thumb img").evaluateAll((images) => images.map((image) => image.getAttribute("src")));
if (new Set(visionThumbnails).size !== 8) {
  throw new Error("five-minute vision nodes do not have distinct thumbnails");
}
await visionPage.getByText("定位球类型待复核", { exact: true }).waitFor();
await visionPage.waitForFunction(() => {
  const video = document.querySelector(".video-stage video");
  return video && video.readyState >= 1 && Math.abs(video.duration - 299.8) < 0.2;
});
const visionVideo = visionPage.locator(".video-stage video");
if (!(await visionVideo.getAttribute("src"))?.includes("test_5min_dense_vision.mp4")) {
  throw new Error("five-minute vision project is not playing the new processed output");
}
await visionPage.locator(".center-play").click();
await visionPage.waitForFunction(() => document.querySelector(".video-stage video")?.currentTime > 0.4);
await visionPage.locator(".center-play").click();
await visionPage.locator(".segment-row").nth(6).click();
const visionSeekTime = await visionVideo.evaluate((video) => video.currentTime);
if (Math.abs(visionSeekTime - 250.267) > 0.7) {
  throw new Error(`five-minute vision timeline seek failed: ${visionSeekTime}`);
}
await assertNoHorizontalOverflow(visionPage, "desktop five-minute vision analysis");
await visionPage.screenshot({ path: path.join(output, "desktop-five-minute-vision.png"), fullPage: true });
await visionPage.close();

const spatialPage = await desktop.newPage();
spatialPage.on("pageerror", (error) => errors.push(`spatial: ${error.message}`));
await spatialPage.goto(url, { waitUntil: "networkidle" });
await spatialPage.getByRole("button", { name: "项目", exact: true }).click();
await spatialPage.getByRole("button", { name: /双机位全场空间结构复盘/ }).click();
await spatialPage.locator(".project-video-row").filter({ hasText: "双机位二维空间分析" }).click();
await spatialPage.getByRole("heading", { name: "时段分析" }).waitFor();
if (await spatialPage.locator(".spatial-window-list button").count() !== 12) {
  throw new Error("spatial report did not load all 12 windows");
}
await spatialPage.waitForFunction(() => {
  const video = document.querySelector(".spatial-video-stage video");
  return video && video.readyState >= 1 && video.duration >= 59;
});
await spatialPage.locator(".spatial-video-stage video").evaluate((video) => video.play());
await spatialPage.waitForFunction(() => document.querySelector(".spatial-video-stage video")?.currentTime > 0.3);
await spatialPage.locator(".spatial-video-stage video").evaluate((video) => video.pause());
await spatialPage.getByRole("button", { name: "跳转到 00:40" }).click();
await spatialPage.getByRole("heading", { name: "00:40–00:45 站位对比" }).waitFor();
const spatialTime = await spatialPage.locator(".spatial-video-stage video").evaluate((video) => video.currentTime);
if (spatialTime < 40 || spatialTime >= 41) throw new Error(`spatial timeline seek failed: ${spatialTime}`);
await assertNoHorizontalOverflow(spatialPage, "desktop spatial analysis");
await spatialPage.screenshot({ path: path.join(output, "desktop-spatial-analysis.png"), fullPage: true });
await spatialPage.close();

const l1Page = await desktop.newPage();
l1Page.on("pageerror", (error) => errors.push(`projection L1: ${error.message}`));
await l1Page.goto(url, { waitUntil: "networkidle" });
await l1Page.getByRole("button", { name: "项目", exact: true }).click();
await l1Page.getByRole("button", { name: /双机位足球与队形复盘/ }).click();
await l1Page.locator(".project-video-row").filter({ hasText: "双机位足球与队形复盘" }).click();
await l1Page.getByRole("heading", { name: "关键事件候选" }).waitFor();
if (await l1Page.locator(".spatial-l1-event-list button").count() !== 6
  || await l1Page.locator(".spatial-event-marker").count() !== 6) {
  throw new Error("projection L1 nodes were not loaded");
}
if (/L1-\d/.test(await l1Page.locator("#spatial-l1-group").innerText())) {
  throw new Error("technical L1 codes are visible in event filters");
}
await l1Page.waitForFunction(() => {
  const video = document.querySelector(".spatial-video-stage video");
  return video && video.readyState >= 1 && Math.abs(video.duration - 60) < 0.1;
});
await l1Page.locator(".spatial-video-stage video").evaluate((video) => video.play());
await l1Page.waitForFunction(() => document.querySelector(".spatial-video-stage video")?.currentTime > 0.3);
await l1Page.locator(".spatial-video-stage video").evaluate((video) => video.pause());
await l1Page.locator(".spatial-event-marker").first().click();
await l1Page.waitForFunction(() => Math.abs(document.querySelector(".spatial-video-stage video")?.currentTime - 36.68) < 0.5);
await l1Page.getByRole("button", { name: "切换视角版", exact: true }).click();
await l1Page.waitForFunction(() => Math.abs(document.querySelector(".spatial-video-stage video")?.currentTime - 36.68) < 0.5);
await l1Page.waitForFunction(() => document.querySelector(".spatial-switch-main canvas")
  ?.getContext("2d")?.getImageData(480, 270, 1, 1).data[0] > 0);
const mainCameraBefore = await l1Page.locator(".spatial-switch-main canvas").evaluate((canvas) => canvas.toDataURL());
await l1Page.getByRole("button", { name: "将机位二切换到主画面" }).click();
await l1Page.waitForTimeout(100);
const mainCameraAfter = await l1Page.locator(".spatial-switch-main canvas").evaluate((canvas) => canvas.toDataURL());
if (mainCameraBefore === mainCameraAfter) throw new Error("switching camera did not change the main view");
const pitchPixels = await l1Page.locator(".spatial-switch-pitch canvas").evaluate((canvas) => {
  const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  let green = 0;
  for (let index = 0; index < pixels.length; index += 16) {
    if (pixels[index + 1] > pixels[index] && pixels[index + 1] > pixels[index + 2]) green += 1;
  }
  return green;
});
if (pitchPixels < 1000) throw new Error("pitch view is blank");
const cropCheck = await l1Page.evaluate(() => {
  const video = document.querySelector(".spatial-video-stage video");
  const canvas = document.querySelector(".spatial-switch-pitch canvas");
  const { data, width } = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
  let green = 0;
  let sampled = 0;
  for (let y = 35; y < 230; y += 4) {
    for (let x = 3; x < 15; x += 3) {
      const i = (y * width + x) * 4;
      if (data[i + 1] > data[i] * 1.15 && data[i + 1] > data[i + 2] * 1.15) green += 1;
      sampled += 1;
    }
  }
  return { decodedWidth: video.videoWidth, greenRatio: green / sampled };
});
if (cropCheck.decodedWidth <= 1280 || cropCheck.greenRatio < 0.75) {
  throw new Error(`sample-aspect-ratio crop regression: ${JSON.stringify(cropCheck)}`);
}
await l1Page.getByRole("button", { name: "原始拼接", exact: true }).click();
await l1Page.waitForFunction(() => Math.abs(document.querySelector(".spatial-video-stage video")?.currentTime - 36.68) < 0.5);
await assertNoHorizontalOverflow(l1Page, "desktop projection L1 analysis");
await l1Page.screenshot({ path: path.join(output, "desktop-projection-l1.png") });
await l1Page.close();

await page.locator('input[type="file"]').setInputFiles(path.join(repoRoot, "soccer_input_dataset/test_5min.mp4"));
const uploadedCard = page.locator(".asset-card").filter({ has: page.getByRole("heading", { name: "test_5min", exact: true }) });
await uploadedCard.getByText("分析完成", { exact: true }).waitFor({ timeout: 30000 });
await uploadedCard.click();
await page.locator(".inspector select").selectOption("project-main");
await page.locator(".inspector").getByRole("button", { name: "添加到所选项目" }).click();

await page.getByRole("button", { name: "项目", exact: true }).click();
await page.getByRole("heading", { name: "项目列表" }).waitFor();
await page.screenshot({ path: path.join(output, "desktop-projects.png"), fullPage: true });

await page.getByRole("button", { name: /五分钟比赛技战术复盘/ }).click();
await page.getByText("点击视频进入技战术分析工作台").waitFor();
const uploadedRow = page.locator(".project-video-row").filter({ has: page.getByText("test_5min.mp4", { exact: true }) });
await uploadedRow.click();
await page.getByRole("heading", { name: "关键事件" }).waitFor();
await page.getByRole("tab", { name: "数据统计" }).waitFor();
const analyzedSource = await page.locator(".video-stage video").getAttribute("src");
if (!analyzedSource?.includes("test_5min_boxes.mp4")) {
  throw new Error(`uploaded video did not use the real processed output: ${analyzedSource}`);
}
if (await page.locator(".segment-row").count() < 10) throw new Error("real report events were not loaded");
await page.locator(".timeline-legend").getByRole("button", { name: /门将事件/ }).click();
if (await page.locator(".segment-list").getByText(/2:56/).count()) {
  throw new Error("stale 2:56 goalkeeper demo event is still visible");
}
await page.locator(".timeline-legend").getByRole("button", { name: "全部", exact: true }).click();
await page.locator(".segment-row").last().click();
const currentTime = await page.locator(".video-stage video").evaluate((video) => video.currentTime);
if (currentTime <= 0) throw new Error(`timeline seek failed: ${currentTime}`);
await page.locator(".center-play").click();
await page.waitForFunction(
  (startTime) => {
    const video = document.querySelector("video");
    return video && !video.paused && video.currentTime > startTime + 0.5;
  },
  currentTime,
  { timeout: 10000 },
);
await page.locator(".center-play").click();
await page.locator(".event-review-bar button.confirm").click();
await page.getByText("已确认", { exact: true }).waitFor();
await page.screenshot({ path: path.join(output, "desktop-analysis.png"), fullPage: true });
await assertNoHorizontalOverflow(page, "desktop analysis");
await page.getByRole("button", { name: "定位球片段", exact: true }).click();
await page.locator(".set-piece-card").first().waitFor();
if (await page.locator(".set-piece-card").count() < 1) throw new Error("set-piece clips were not loaded");
const setPieceSource = await page.locator(".set-piece-card video").first().getAttribute("src");
if (!setPieceSource?.includes("evidence")) throw new Error(`invalid set-piece clip source: ${setPieceSource}`);
await page.waitForFunction(() => {
  const clip = document.querySelector(".set-piece-card video");
  return clip && clip.readyState >= 1 && clip.duration >= 9 && clip.duration <= 10.1;
});
await page.screenshot({ path: path.join(output, "desktop-set-pieces.png"), fullPage: true });

const longPage = await desktop.newPage();
longPage.on("pageerror", (error) => errors.push(`long video: ${error.message}`));
await longPage.goto(url, { waitUntil: "networkidle" });
await longPage.getByRole("button", { name: "项目", exact: true }).click();
await longPage.getByRole("button", { name: /曼联 vs 莱斯特 40分钟技战术复盘/ }).click();
await longPage.locator(".project-video-row").filter({ hasText: "SoccerNet 40分钟长视频测试" }).click();
await longPage.getByRole("heading", { name: "关键事件" }).waitFor();
await longPage.getByLabel("定位球分类次数").getByText("任意球：5", { exact: true }).waitFor();
await longPage.getByLabel("定位球分类次数").getByText("角球：1", { exact: true }).waitFor();
const longVideo = longPage.locator("video");
const longVideoSource = await longVideo.getAttribute("src");
if (!longVideoSource?.includes("03_tactical_red_blue_v2_web.mp4")) {
  throw new Error(`long project did not use the processed output: ${longVideoSource}`);
}
await longPage.waitForFunction(() => {
  const video = document.querySelector("video");
  return video && video.readyState >= 1 && video.duration >= 2399;
});
if (await longPage.locator(".segment-row").count() !== 111) {
  throw new Error("long project did not load all 111 review nodes");
}
await longPage.locator(".center-play").click();
await longPage.waitForFunction(() => {
  const video = document.querySelector("video");
  return video && !video.paused && video.currentTime > 0.5;
}, null, { timeout: 10000 });
await longPage.locator(".center-play").click();
await longPage.screenshot({ path: path.join(output, "desktop-long-analysis.png"), fullPage: true });
await longPage.close();

const mobile = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
const mobilePage = await mobile.newPage();
mobilePage.on("pageerror", (error) => errors.push(`mobile: ${error.message}`));
await mobilePage.goto(url, { waitUntil: "networkidle" });
await mobilePage.getByRole("heading", { name: "比赛素材库" }).waitFor();
await assertNoHorizontalOverflow(mobilePage, "mobile assets");
await mobilePage.screenshot({ path: path.join(output, "mobile-assets.png"), fullPage: true });
await mobilePage.getByRole("button", { name: "打开导航" }).click();
await mobilePage.getByRole("button", { name: "项目", exact: true }).click();
await mobilePage.waitForFunction(() => !document.querySelector(".sidebar")?.classList.contains("is-open"));
await mobilePage.waitForTimeout(250);
await mobilePage.getByRole("heading", { name: "项目列表" }).waitFor();
await assertNoHorizontalOverflow(mobilePage, "mobile projects");
await mobilePage.screenshot({ path: path.join(output, "mobile-projects.png"), fullPage: true });
await mobilePage.getByRole("button", { name: /双机位全场空间结构复盘/ }).click();
await mobilePage.locator(".project-video-row").filter({ hasText: "双机位二维空间分析" }).click();
await mobilePage.getByRole("heading", { name: "时段分析" }).waitFor();
await assertNoHorizontalOverflow(mobilePage, "mobile spatial analysis");
await mobilePage.screenshot({ path: path.join(output, "mobile-spatial-analysis.png"), fullPage: true });

const mobileVision = await mobile.newPage();
mobileVision.on("pageerror", (error) => errors.push(`mobile five-minute vision: ${error.message}`));
await mobileVision.goto(url, { waitUntil: "networkidle" });
await mobileVision.getByRole("button", { name: "打开导航" }).click();
await mobileVision.getByRole("button", { name: "项目", exact: true }).click();
await mobileVision.getByRole("button", { name: /五分钟多帧视觉复核/ }).click();
await mobileVision.locator(".project-video-row").filter({ hasText: "五分钟多帧视觉复核测试" }).click();
await mobileVision.getByRole("heading", { name: "关键事件" }).waitFor();
await assertNoHorizontalOverflow(mobileVision, "mobile five-minute vision analysis");
await mobileVision.screenshot({ path: path.join(output, "mobile-five-minute-vision.png"), fullPage: true });
await mobileVision.close();

const mobileL1 = await mobile.newPage();
mobileL1.on("pageerror", (error) => errors.push(`mobile projection L1: ${error.message}`));
await mobileL1.goto(url, { waitUntil: "networkidle" });
await mobileL1.getByRole("button", { name: "打开导航" }).click();
await mobileL1.getByRole("button", { name: "项目", exact: true }).click();
await mobileL1.getByRole("button", { name: /双机位足球与队形复盘/ }).click();
await mobileL1.locator(".project-video-row").filter({ hasText: "双机位足球与队形复盘" }).click();
await mobileL1.getByRole("button", { name: "切换视角版", exact: true }).click();
await mobileL1.waitForFunction(() => document.querySelector(".spatial-switch-pitch canvas")
  ?.getContext("2d")?.getImageData(168, 136, 1, 1).data[1] > 0);
await assertNoHorizontalOverflow(mobileL1, "mobile switchable camera view");
await mobileL1.screenshot({ path: path.join(output, "mobile-switchable-camera.png") });
await mobileL1.close();

async function verifyLayered(context, mobileView) {
  const layered = await context.newPage();
  layered.on("pageerror", (error) => errors.push(`layered: ${error.message}`));
  await layered.goto(url, { waitUntil: "networkidle" });
  if (mobileView) await layered.getByRole("button", { name: "打开导航" }).click();
  await layered.getByRole("button", { name: "项目", exact: true }).click();
  await layered.getByRole("button", { name: /五分钟分层技战术复盘/ }).click();
  await layered.locator(".project-video-row").filter({ hasText: "五分钟分层技战术分析" }).click();
  await layered.getByRole("heading", { name: "关键事件" }).waitFor();
  await layered.waitForFunction(() => document.querySelectorAll(".segment-row").length === 17);
  if (await layered.locator(".video-analysis-badge, .event-overlay").count()) throw new Error("boxes-only video has overlays");
  await layered.waitForFunction(() => {
    const v = document.querySelector(".video-stage video");
    return v?.readyState >= 2 && v.duration > 299 && v.videoWidth === 1280;
  });
  await layered.locator(".timeline-legend").getByRole("button", { name: /门将事件/ }).click();
  await layered.locator(".segment-row").first().click();
  await layered.waitForFunction(() => Math.abs(document.querySelector(".video-stage video").currentTime - 208.03) < 0.2);
  await layered.locator(".center-play").click();
  await layered.waitForFunction(() => document.querySelector(".video-stage video").currentTime > 208.7);
  await layered.locator(".center-play").click();
  const pixels = await layered.locator(".video-stage video").evaluate((video) => {
    const canvas = document.createElement("canvas"); canvas.width = 32; canvas.height = 18;
    const ctx = canvas.getContext("2d"); ctx.drawImage(video, 0, 0, 32, 18);
    return [...ctx.getImageData(0, 0, 32, 18).data].filter((v, i) => i % 4 !== 3 && v > 20).length;
  });
  if (pixels < 100) throw new Error("layered video decoded blank pixels");
  await layered.getByText("审核片段与逐帧证据", { exact: false }).click();
  await layered.waitForFunction(() => {
    const v = document.querySelector(".evidence-details video");
    return v?.readyState >= 1 && v.duration >= 9 && v.duration <= 10.1;
  });
  await layered.locator(".evidence-details video").evaluate(async (v) => { v.muted = true; await v.play(); });
  await layered.waitForFunction(() => document.querySelector(".evidence-details video").currentTime > 0.3);
  await layered.locator(".evidence-details video").evaluate((v) => v.pause());
  if (await layered.locator(".evidence-filmstrip button").count() !== 41) throw new Error("missing 41-frame evidence");
  await layered.locator(".evidence-filmstrip button").first().click();
  await layered.waitForFunction(() => Math.abs(document.querySelector(".video-stage video").currentTime - 203.03) < 0.2);
  await assertNoHorizontalOverflow(layered, "layered L1 evidence");
  await layered.screenshot({ path: path.join(output, `${mobileView ? "mobile" : "desktop"}-layered-l1.png`), fullPage: true });
  await layered.getByRole("tab", { name: "数据统计" }).click();
  await layered.locator(".layer-counts").waitFor();
  await assertNoHorizontalOverflow(layered, "layered L2");
  await layered.getByRole("tab", { name: "进攻组织" }).click();
  await layered.locator(".layer-sequences article").first().waitFor();
  await layered.locator(".layer-sequences button").first().click();
  await layered.getByRole("tab", { name: "队形空间" }).click();
  await layered.locator(".layer-sequences article").first().waitFor();
  await assertNoHorizontalOverflow(layered, "layered L4");
  await layered.screenshot({ path: path.join(output, `${mobileView ? "mobile" : "desktop"}-layered-l4.png`), fullPage: true });
  await layered.close();
}
await verifyLayered(desktop, false);
await verifyLayered(mobile, true);

async function verifyAssistantReview(context, mobileView) {
  const reviewed = await context.newPage();
  reviewed.on("pageerror", (error) => errors.push(`assistant review: ${error.message}`));
  await reviewed.goto(url, { waitUntil: "networkidle" });
  if (mobileView) await reviewed.getByRole("button", { name: "打开导航" }).click();
  await reviewed.getByRole("button", { name: "项目", exact: true }).click();
  await reviewed.getByRole("button", { name: /五分钟助手复核对照测试/ }).click();
  await reviewed.locator(".project-video-row").filter({ hasText: "五分钟助手切片复核版" }).click();
  await reviewed.getByLabel("助手复核意见").waitFor();
  if (await reviewed.locator(".segment-row").count() !== 18) throw new Error("assistant review nodes missing");
  await reviewed.getByLabel("定位球分类次数").getByText("界外球：1", { exact: true }).waitFor();
  await reviewed.getByLabel("定位球分类次数").getByText("角球：0", { exact: true }).waitFor();
  await reviewed.getByLabel("助手复核意见").selectOption("排除");
  if (await reviewed.locator(".segment-row").count() !== 3) throw new Error("rejected review filtering failed");
  await reviewed.getByLabel("助手复核意见").selectOption("待核实");
  if (await reviewed.locator(".segment-row").count() !== 1) throw new Error("uncertain review filtering failed");
  await reviewed.getByLabel("助手复核意见").selectOption("全部意见");
  await reviewed.locator(".segment-row").filter({ hasText: "白队门将接回传后大脚出球" }).click();
  await reviewed.waitForFunction(() => Math.abs(document.querySelector(".video-stage video")?.currentTime - 208.033) < 0.2);
  await reviewed.getByText(/智能复核.*非人工专家标注/).waitFor();
  if (await reviewed.locator(".video-analysis-badge, .event-overlay").count()) throw new Error("unexpected burned UI overlay");
  await reviewed.locator(".center-play").click();
  await reviewed.waitForFunction(() => document.querySelector(".video-stage video")?.currentTime > 208.6);
  await reviewed.locator(".center-play").click();
  await reviewed.getByText(/本次实际查验的来源帧/).click();
  await reviewed.locator(".evidence-filmstrip img").first().waitFor();
  await assertNoHorizontalOverflow(reviewed, "assistant evidence");
  await reviewed.evaluate(() => window.scrollTo(0, 0));
  await reviewed.screenshot({ path: path.join(output, `${mobileView ? "mobile" : "desktop"}-assistant-review.png`), fullPage: true });
  await reviewed.getByRole("tab", { name: "数据统计" }).click();
  await reviewed.getByRole("heading", { name: "本次切片复核统计" }).waitFor();
  const shootingRow = reviewed.locator(".layer-counts tbody tr").filter({ hasText: "射门" });
  if ((await shootingRow.locator("td").allTextContents()).join(",") !== "4,0,0,0") throw new Error("corrected shots still counted");
  await reviewed.getByRole("button", { name: "定位球片段", exact: true }).click();
  await reviewed.locator(".set-piece-card").first().waitFor();
  if (await reviewed.locator(".set-piece-card").count() !== 1) throw new Error("rejected corner remains in set pieces");
  await reviewed.waitForFunction(() => document.querySelector(".set-piece-card video")?.duration >= 9.9);
  await assertNoHorizontalOverflow(reviewed, "assistant set pieces");
  await reviewed.close();
}
await verifyAssistantReview(desktop, false);
await verifyAssistantReview(mobile, true);

async function verifyDeepReview(context, mobileView) {
  const review = await context.newPage();
  review.on("pageerror", (error) => errors.push(`deep review: ${error.message}`));
  await review.goto(url, { waitUntil: "networkidle" });
  if (mobileView) await review.getByRole("button", { name: "打开导航" }).click();
  await review.getByRole("button", { name: "项目", exact: true }).click();
  await review.getByRole("button", { name: /五分钟战术深度复盘/ }).click();
  await review.locator(".project-video-row").filter({ hasText: "五分钟战术深度复盘" }).click();
  await review.getByRole("heading", { name: "关键事件" }).waitFor();
  const assertReadableLabels = async () => {
    if (/\bL[1-4](?:\b|事件|分析)/.test(await review.locator("body").innerText())) {
      throw new Error("Internal analysis level codes are visible in the UI");
    }
    const tabs = await review.getByRole("tablist", { name: "技战术分析分类" }).getByRole("tab").allTextContents();
    if (tabs.join(",") !== "关键事件,数据统计,进攻组织,队形空间") throw new Error("Analysis labels are inconsistent");
  };
  await assertReadableLabels();
  if (await review.locator(".segment-row").count() !== 18) throw new Error("deep review L1 missing");
  const assertPanelAlignment = async (label) => {
    if (mobileView) return;
    await review.waitForFunction(() => {
      const panel = document.querySelector(".segments-panel");
      const divider = document.querySelector(".analysis-commentary > .ai-review-detail");
      return panel && divider && Math.abs(panel.getBoundingClientRect().bottom - divider.getBoundingClientRect().top) < 2;
    });
    const scrolls = await review.evaluate(() => {
      const list = document.querySelector(".segments-panel .layer-analysis, .segments-panel .segment-list");
      list.scrollTop = list.scrollHeight;
      const scrollable = list.scrollHeight > list.clientHeight && list.scrollTop > 0;
      list.scrollTop = 0;
      return scrollable;
    });
    if (!scrolls) throw new Error(`${label}: the clip list must remain independently scrollable`);
  };
  await assertPanelAlignment("L1 desktop");
  await review.getByRole("tab", { name: "进攻组织" }).click();
  await review.getByRole("heading", { name: "有球组织与进攻过程" }).waitFor();
  if (await review.locator(".layer-sequences article").count() !== 3) throw new Error("L3 sequence rules missing");
  await assertPanelAlignment("L3 desktop");
  if (!mobileView) {
    await review.setViewportSize({ width: 2555, height: 1393 });
    await assertPanelAlignment("L3 wide desktop");
    await review.screenshot({ path: path.join(output, "wide-aligned-clips.png"), fullPage: true });
    await review.setViewportSize({ width: 1440, height: 1000 });
    await assertPanelAlignment("L3 resized desktop");
  }
  await review.getByRole("button", { name: "播放蓝队门将参与后场循环组织", exact: true }).click();
  await review.waitForFunction(() => Math.abs(document.querySelector(".video-stage video")?.currentTime - 242.567) < 0.25);
  await review.getByRole("region", { name: "战术评语" }).getByText("复盘建议", { exact: true }).waitFor();
  await review.getByText(/复核依据 · 11 张时序画面/).click();
  await review.locator(".tactical-commentary .evidence-filmstrip img").first().scrollIntoViewIfNeeded();
  await review.waitForFunction(() => {
    const image = document.querySelector(".tactical-commentary .evidence-filmstrip img");
    return image?.complete && image.naturalWidth > 0;
  });
  await review.getByText(/完整上下文片段/).click();
  const clip = review.locator(".tactical-commentary video");
  await clip.evaluate((video) => video.load());
  await review.waitForFunction(() => document.querySelector(".tactical-commentary video")?.duration > 25);
  await assertPanelAlignment("expanded evidence does not stretch the clip panel");
  if (/codex/i.test(await review.locator("body").innerText())) throw new Error("provider name visible");
  await review.locator(".center-play").click();
  await review.waitForFunction(() => document.querySelector(".video-stage video")?.currentTime > 243.2);
  await review.locator(".center-play").click();
  await assertNoHorizontalOverflow(review, "deep review organization");
  await review.evaluate(() => window.scrollTo(0, 0));
  await review.screenshot({ path: path.join(output, `${mobileView ? "mobile" : "desktop"}-deep-review.png`), fullPage: true });
  await review.getByRole("tab", { name: "队形空间" }).click();
  await review.getByLabel("战术复核筛选").selectOption("采纳");
  if (await review.locator(".layer-sequences article").count() !== 3) throw new Error("L4 accepted filtering failed");
  await review.getByLabel("战术复核筛选").selectOption("未采纳");
  if (await review.locator(".layer-sequences article").count() !== 3) throw new Error("L4 rejected filtering failed");
  await review.getByLabel("战术复核筛选").selectOption("待核实");
  if (await review.locator(".layer-sequences article").count() !== 2) throw new Error("L4 uncertain filtering failed");
  await assertNoHorizontalOverflow(review, "deep review spatial");
  await assertReadableLabels();
  await review.getByRole("button", { name: "定位球片段", exact: true }).click();
  if (/\bL[1-4]\b/.test(await review.locator("body").innerText())) throw new Error("Set piece heading still shows level codes");
  await review.close();
}
await verifyDeepReview(desktop, false);
await verifyDeepReview(mobile, true);

await desktop.close();
await mobile.close();
await browser.close();

if (errors.length) {
  throw new Error(`Browser errors:\n${errors.join("\n")}`);
}

console.log("Playwright verification passed");
console.log(`Screenshots: ${output}`);
