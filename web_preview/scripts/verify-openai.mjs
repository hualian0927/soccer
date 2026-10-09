import { chromium } from "@playwright/test";
import { mkdir, rm } from "node:fs/promises";
import { createReadStream } from "node:fs";
import path from "node:path";

const root = process.cwd();
const url = process.env.PREVIEW_URL || "http://127.0.0.1:4173";
const resultId = process.env.REVIEW_RESULT_ID || "test5-openai-v1";
const title = process.env.REVIEW_PROJECT_TITLE || "五分钟智能视觉复盘";
const result = await fetch(`${url}/api/demo-results/${resultId}`).then((r) => {
  if (!r.ok) throw new Error(`Result endpoint: ${r.status}`);
  return r.json();
});
if (!result.events.length || !result.layers.L2 || !result.pipeline.includes("自动时序")) throw new Error("Missing API review result");
for (const event of result.events) {
  if (!Number.isFinite(event.time) || event.time < 0 || event.time > 300) throw new Error("Invalid event timestamp");
  if (event.publishedForStatistics && event.visionStatus !== "visual_supported_candidate") throw new Error("Ungated accepted event");
}
const range = await fetch(`${url}${result.videoUrl}`, {headers: {Range: "bytes=0-99"}});
if (range.status !== 206) throw new Error("Video range support missing");
await range.arrayBuffer();
if (process.argv.includes("--upload-cache")) {
  if (result.reviewWarnings.length) throw new Error("Refusing cache upload test with incomplete review");
  const report = await fetch(`${url}${result.reportUrl}`).then((r) => r.json());
  if (!report.reviewMetadata.boundary_transition_audit) throw new Error("Cache policy not ready");
  const response = await fetch(`${url}/api/analysis`, {method: "POST", duplex: "half",
    headers: {"Content-Type": "video/mp4", "X-File-Name": "openai-cache-test.mp4"},
    body: createReadStream(path.join(root, "../soccer_input_dataset/test_5min.mp4"))});
  const job = await response.json();
  if (!response.ok || !/^[a-f0-9-]+$/.test(job.id)) throw new Error("Cache upload failed");
  let completed = false;
  for (let i = 0; i < 30; i++) {
    const status = await fetch(`${url}/api/analysis/${job.id}`).then((r) => r.json());
    if (status.status === "completed") {
      if (!status.result.cached || !status.result.pipeline.includes("自动时序")) throw new Error("Wrong upload pipeline");
      completed = true;
      break;
    }
    if (status.progress >= 15 || status.status === "failed") throw new Error("Upload did not reuse the expected cache");
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  if (!completed) throw new Error("Upload cache timeout");
  await rm(path.join(root, "runtime/jobs", job.id), {recursive: true, force: true});
  console.log("Upload reused the OpenAI result; temporary test upload removed.");
}
const output = path.join(root, "test-results");
await mkdir(output, {recursive: true});
const browser = await chromium.launch({headless: true,
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH || path.join(process.env.HOME, ".cache/ms-playwright/chromium-1234/chrome-linux64/chrome"),
  env: {...process.env, LD_LIBRARY_PATH: `${root}/test-runtime/root/usr/lib/x86_64-linux-gnu:${process.env.LD_LIBRARY_PATH || ""}`}});
const errors = [];
try {
  for (const [name, viewport] of [["desktop", {width: 1440, height: 1000}], ["mobile", {width: 390, height: 844}]]) {
    const page = await browser.newPage({viewport});
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(url, {waitUntil: "networkidle"});
    if (name === "mobile") {
      const toggle = page.getByRole("button", {name: "打开导航"});
      if (await toggle.count()) await toggle.click();
    }
    await page.getByRole("button", {name: "项目", exact: true}).click();
    await page.getByRole("button", {name: new RegExp(title)}).click();
    await page.locator(".project-video-row").filter({hasText: title}).click();
    await page.getByRole("heading", {name: "关键事件", exact: true}).waitFor();
    await page.waitForFunction(() => document.querySelector(".video-stage video")?.readyState >= 2);
    const video = page.locator(".video-stage video");
    await video.evaluate(async (v) => {v.muted = true; await v.play();});
    await page.waitForFunction(() => document.querySelector(".video-stage video")?.currentTime > .5);
    await video.evaluate((v) => v.pause());
    await page.locator(".segment-row").first().click();
    const time = await video.evaluate((v) => v.currentTime);
    if (Math.abs(time - result.events[0].time) > 1) throw new Error(`Seek mismatch: ${time}`);
    await page.waitForFunction(() => {const v = document.querySelector(".video-stage video"); return v && !v.seeking && v.readyState >= 2;});
    await page.getByRole("combobox", {name: "助手复核意见"}).selectOption("待核实");
    if (await page.locator(".segment-row").count() !== result.events.filter((e) => e.reviewDecision === "uncertain").length) throw new Error("Review filter mismatch");
    await page.getByRole("combobox", {name: "助手复核意见"}).selectOption("全部意见");
    await page.screenshot({path: path.join(output, `${resultId}-${name}.png`), fullPage: true});
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    if (overflow) throw new Error(`${name} overflow`);
    for (const label of ["数据统计", "进攻组织", "队形空间"]) {
      await page.getByRole("tab", {name: label, exact: true}).click();
      await page.getByRole("heading", {name: label, exact: true}).waitFor();
    }
    await page.close();
  }
  if (errors.length) throw new Error(errors.join("\n"));
  console.log(JSON.stringify({events: result.events.length, accepted: result.events.filter((e) => e.publishedForStatistics).length,
    videoRange: true, desktop: "passed", mobile: "passed", filters: "passed", layers: "passed"}));
} finally {
  await browser.close();
}
