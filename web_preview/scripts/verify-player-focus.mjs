import { chromium, expect } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";

const root=process.cwd(), url=process.env.PREVIEW_URL || "http://127.0.0.1:4173";
const output=path.join(root,"test-results");await mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,
  executablePath:process.env.PLAYWRIGHT_CHROMIUM_PATH || path.join(process.env.HOME,".cache/ms-playwright/chromium-1234/chrome-linux64/chrome"),
  env:{...process.env,LD_LIBRARY_PATH:`${root}/test-runtime/root/usr/lib/x86_64-linux-gnu:${process.env.LD_LIBRARY_PATH || ""}`}});
const errors=[];
try {
  for(const [name,viewport] of [["desktop",{width:1440,height:1000}],["mobile",{width:390,height:844}]]) {
    const page=await browser.newPage({viewport});page.on("pageerror",e=>errors.push(e.message));
    await page.goto(url,{waitUntil:"networkidle"});
    if(name==="mobile")await page.getByRole("button",{name:"打开导航"}).click();
    await page.getByRole("button",{name:"项目",exact:true}).click();
    await page.getByRole("button",{name:/五分钟球员编号与个人复盘/}).click();
    await page.locator(".project-video-row").filter({hasText:"五分钟球员编号与个人复盘"}).click();
    await page.getByRole("heading",{name:"球员片段分析",exact:true}).waitFor();
    await expect(page.locator(".player-id-overlay")).toHaveCount(0);
    const video=page.locator(".video-stage video");
    await page.waitForFunction(()=>document.querySelector(".video-stage video")?.readyState>=2);
    await video.evaluate(v=>{v.currentTime=145;v.pause();});
    await page.getByRole("checkbox",{name:"球员编号"}).check();
    await page.waitForFunction(()=>document.querySelectorAll(".player-id").length>3);
    await page.locator(".player-id").first().click();
    await expect(page.locator(".player-id.selected")).toHaveCount(1);
    await page.locator(".focus-examples button").first().click();
    await page.getByRole("button",{name:"生成片段分析",exact:true}).click();
    await page.locator(".focus-result").waitFor({timeout:120000});
    await expect(page.locator(".focus-result")).toContainText("技战术解读");
    const clip=page.locator(".focus-result video");
    await expect(clip).toHaveAttribute("src",/api\/media/);
    await clip.evaluate(async v=>{v.muted=true;await v.play();});
    await page.waitForFunction(()=>document.querySelector(".focus-result video")?.currentTime>.2);
    await clip.evaluate(v=>v.pause());
    await page.locator(".video-stage").scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(output,`player-focus-${name}.png`),fullPage:true});
    await page.locator(".focus-form").scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(output,`player-focus-${name}-detail.png`)});
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw Error(`${name} overflow`);
    await page.getByRole("checkbox",{name:"球员编号"}).uncheck();
    await expect(page.locator(".player-id-overlay")).toHaveCount(0);
    await expect(page.locator(".focus-result")).toHaveCount(0);
    await page.close();
  }
  if(errors.length)throw Error(errors.join("\n"));
  const invalid=await fetch(`${url}/api/player-review`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({playerId:1,start:0,end:1,trackingPath:"../outside.json"})});
  if(invalid.status!==400)throw Error("Path validation failed");
  console.log("Player focus: desktop/mobile, hidden default, selection, reviewed export, hide, path rejection passed.");
} finally {await browser.close();}
