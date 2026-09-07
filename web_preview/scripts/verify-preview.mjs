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
await page.screenshot({ path: path.join(output, "desktop-assets.png"), fullPage: true });

await page.locator('input[type="file"]').setInputFiles(path.join(repoRoot, "soccer_input_dataset/test_5min.mp4"));
const uploadedCard = page.locator(".asset-card").filter({ has: page.getByRole("heading", { name: "test_5min", exact: true }) });
await uploadedCard.getByText("分析完成", { exact: true }).waitFor({ timeout: 30000 });
await uploadedCard.click();
await page.locator(".inspector").getByRole("button", { name: "添加到所选项目" }).click();

await page.getByRole("button", { name: "项目", exact: true }).click();
await page.getByRole("heading", { name: "项目列表" }).waitFor();
await page.screenshot({ path: path.join(output, "desktop-projects.png"), fullPage: true });

await page.getByRole("button", { name: /五分钟比赛技战术复盘/ }).click();
await page.getByText("点击视频进入 L1 事件分析工作台").waitFor();
const uploadedRow = page.locator(".project-video-row").filter({ has: page.getByText("test_5min.mp4", { exact: true }) });
await uploadedRow.click();
await page.getByRole("heading", { name: "L1 事件节点" }).waitFor();
await page.getByLabel("定位球分类次数").getByText("角球：2", { exact: true }).waitFor();
const analyzedSource = await page.locator("video").getAttribute("src");
if (!analyzedSource?.includes("test_5min_tactical_base_web.mp4")) {
  throw new Error(`uploaded video did not use the real processed output: ${analyzedSource}`);
}
if (await page.locator(".segment-row").count() < 10) throw new Error("real report events were not loaded");
await page.locator(".timeline-legend").getByRole("button", { name: /门将事件/ }).click();
await page.locator(".segment-row").first().getByText("3:28–3:30", { exact: true }).waitFor();
if (await page.locator(".segment-list").getByText(/2:56/).count()) {
  throw new Error("stale 2:56 goalkeeper demo event is still visible");
}
await page.locator(".timeline-legend").getByRole("button", { name: "全部", exact: true }).click();
await page.locator(".segment-row").last().click();
const currentTime = await page.locator("video").evaluate((video) => video.currentTime);
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
if (!setPieceSource?.includes("set_piece_clips")) throw new Error(`invalid set-piece clip source: ${setPieceSource}`);
await page.screenshot({ path: path.join(output, "desktop-set-pieces.png"), fullPage: true });

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

await desktop.close();
await mobile.close();
await browser.close();

if (errors.length) {
  throw new Error(`Browser errors:\n${errors.join("\n")}`);
}

console.log("Playwright verification passed");
console.log(`Screenshots: ${output}`);
