import {chromium,expect} from "@playwright/test";
import {mkdir} from "node:fs/promises";
import path from "node:path";

const root=process.cwd(),url=process.env.PREVIEW_URL || "http://127.0.0.1:4173";
await mkdir("test-results",{recursive:true});
const browser=await chromium.launch({headless:true,
  executablePath:path.join(process.env.HOME,".cache/ms-playwright/chromium-1234/chrome-linux64/chrome"),
  env:{...process.env,LD_LIBRARY_PATH:`${root}/test-runtime/root/usr/lib/x86_64-linux-gnu:${process.env.LD_LIBRARY_PATH || ""}`}});
const errors=[];
try {
  for(const [name,viewport] of [["desktop",{width:1440,height:1000}],["mobile",{width:390,height:844}]]) {
    const page=await browser.newPage({viewport});
    page.on("pageerror",e=>errors.push(e.message));
    await page.goto(url,{waitUntil:"networkidle"});
    if(name==="mobile")await page.getByRole("button",{name:"打开导航"}).click();
    await page.getByRole("button",{name:"项目",exact:true}).click();
    await page.getByRole("button",{name:/五分钟阵型可视化复盘/}).click();
    await page.locator(".project-video-row").filter({hasText:"五分钟阵型可视化复盘"}).click();
    await page.getByRole("tab",{name:"队形空间",exact:true}).click();
    await expect(page.locator(".formation-evidence")).toHaveCount(0);
    const open=page.getByRole("button",{name:/查看阵型分布/}).first();
    await open.click();
    await page.waitForFunction(()=>document.querySelector(".formation-media img")?.naturalWidth>100);
    await expect(page.locator(".video-stage > video")).toHaveJSProperty("paused",true);
    await page.locator(".video-stage").scrollIntoViewIfNeeded();
    await page.screenshot({path:`test-results/formation-${name}.png`,fullPage:true});
    await page.screenshot({path:`test-results/formation-${name}-viewport.png`});
    const colors=await page.locator(".formation-media img").evaluate(img=>{
      const canvas=document.createElement("canvas");canvas.width=32;canvas.height=18;
      const ctx=canvas.getContext("2d");ctx.drawImage(img,0,0,32,18);
      return new Set(Array.from(ctx.getImageData(0,0,32,18).data)).size;
    });
    if(colors<20)throw Error("Blank formation frame");
    await page.getByRole("button",{name:"原始画面",exact:true}).click();
    await expect(page.locator(".formation-media img")).toHaveAttribute("src",/original/);
    await page.getByRole("button",{name:"战术片段",exact:true}).click();
    await page.locator(".formation-media video").evaluate(async v=>{v.muted=true;await v.play();});
    await page.waitForFunction(()=>document.querySelector(".formation-media video")?.currentTime>.2);
    await page.getByRole("tab",{name:"关键事件",exact:true}).click();
    await expect(page.locator(".formation-evidence")).toHaveCount(0);
    await page.getByRole("tab",{name:"队形空间",exact:true}).click();await open.click();
    await page.getByRole("button",{name:"关闭阵型标注",exact:true}).click();
    await expect(page.locator(".formation-evidence")).toHaveCount(0);
    await open.click();await page.getByRole("button",{name:"前进五秒",exact:true}).click();
    await expect(page.locator(".formation-evidence")).toHaveCount(0);
    await page.getByRole("tab",{name:"关键事件",exact:true}).click();
    const count=await page.locator(".insight-strip > div").last().locator("strong").textContent();
    await page.getByRole("button",{name:"拒绝",exact:true}).click();
    await page.getByRole("tab",{name:"数据统计",exact:true}).click();
    await expect(page.locator(".layer-analysis")).toContainText("随人工确认与拒绝同步更新");
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw Error(`${name} overflow`);
    console.log(name,"image pixels",colors,"initial accepted",count);
    await page.close();
  }
  if(errors.length)throw Error(errors.join("\n"));
  console.log("Formation display, original frame, clip playback, tab/close/seek hiding, statistics, desktop/mobile passed.");
}finally{await browser.close();}
