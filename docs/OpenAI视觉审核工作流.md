# OpenAI 时序视觉审核工作流

更新：2026-09-30。

## 目标与边界

将此前助手参与的“取证、判断、补证、校验、输出”固化为可重复的 API 流程。
指定模型为 `gpt-5.6-sol`，不自动切换其他模型。并非只替换旧 DeepSeek 请求中的模型名称。
旧审核样例保留，新结果独立保存。当前不能承诺已达到旧版本或教练审核的准确率。

本轮不重训检测器，不替换跟踪数据。主视频仍是原画面加人物与足球检测框；修改的是审核、事件分类、时序证据和技战术评语。

## 流程

1. 读取本地技战术报告的算法候选，不把旧助手审核结论提供给模型。
2. 初始取候选前后各 5 秒的 21 张原视频画面，保留原视频时间戳。
3. 另按 30 秒窗口扫描全片，每窗提取一个最相关事件候选，补充算法漏检。
4. Responses API 接收完整图片内容、球队颜色映射和审核约束，按 JSON Schema 返回结果。
5. 模型可请求延长窗口、加密抽帧或归一化目标区域裁剪；最多补证两轮，且保留前后文。
6. 门将采纳候选再独立审核一次，不将初审结论附给复审；类型或球队意见不一致则待核实。
7. 校验引用帧、时间区间、重启动作和门将身份/来球/接触证据；失败或不足不进入采纳统计。
   出界事件另用扩展窗口复审“是否发生了新的越界”，边线外持球等待不能再计一次出界。
   门将脚下传球允许由连续前后画面支持类别、但精确触球时刻未知；手部扑救仍要求触球及来球证据。
8. 基于已采纳事件生成进攻组织规则候选，再审核其连续画面；队形空间只给可见区域定性分析。
9. 输出原有前端可以读取的 `layered_analysis.json`、中文报告和可回看的审核片段。

30 秒粗扫不是逐帧检测，每窗只提取一个事件，不能据此宣称全场事件完整覆盖。
精确空间结论一律不发布；可见性门槛通过也不代表场地标定准确。
出界只支持出界，不能单独确认角球、界外球、球门球等具体重启方式。

## 代码

| 文件 | 作用 |
| --- | --- |
| `tactical_analysis/openai_review.py` | 官方 API 客户端、结构化契约、证据门控、预算与调用账本 |
| `workflows/review/run_openai_tactical_review.py` | 抽帧、动态补证、独立复审、事件去重、组织/空间复核、报告发布 |
| `tests/test_openai_tactical_review.py` | 时间和帧引用校验、门将/定位球门控、费用约束、模型与拒绝状态等测试 |
| `web_preview/server/local-analysis.js` | 网页上传后选择同一 OpenAI 审核入口、读取独立结果 |

## 密钥与环境

API 密钥只由后端环境变量 `OPENAI_API_KEY` 或终端隐藏输入提供。
不要放入前端 `VITE_*` 变量、源码、命令参数、文档或 Git。
已经在聊天中公开的密钥应撤销后重新创建。

```bash
conda activate sports
pip install 'jsonschema>=4.23,<5'
read -rs -p 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
```

## 本地命令

```bash
python -m workflows.review.run_openai_tactical_review \
  --bundle soccer_input_dataset/outputs/test_5min_layered_v2/layered_analysis.json \
  --output-dir soccer_input_dataset/outputs/test_5min_openai_v1 \
  --baseline soccer_input_dataset/outputs/test_5min_layered_v4/layered_analysis.json \
  --max-calls 80 --max-usd 8
```

也可省略环境变量并添加 `--prompt-key`，通过隐藏终端输入仅供本次进程使用。
添加 `--cached-only` 可不带密钥、零网络调用重建已有完整审核报告；缓存缺失会中止，不覆盖已有报告。
新素材使用 `--report-json <tactical_analysis_report.json>`，而不是 `--bundle`。
`--team-left` 和 `--team-right` 定义轨迹队伍到服装的映射，必须与上游身份配置一致。
五分钟样例默认蓝衣队（黄绿色门将）与白衣队；不应不加修改地用于不同球衣比赛。

## 网页调用

在同一个已设置 `OPENAI_API_KEY` 的终端中：

```bash
cd web_preview
npm run dev -- --host 127.0.0.1
```

前端上传 -> WSL 后端检测跟踪 -> 身份修正 -> 检测框视频 -> 算法报告 -> 本脚本 -> 前端显示。
有 OpenAI 密钥时优先走该路径；无 OpenAI 密钥时保留原流程，不会假装已经调用 OpenAI。
可设置 `OPENAI_REVIEW_MAX_USD` 调整每个任务预算，默认 8 美元。每个新上传任务分别计算费用。
完全相同的五分钟原素材，仅在 OpenAI 缓存版本匹配且无失败/预算中断时复用，避免重复付费。

## 输出与审计

- `layered_analysis.json`：事件、数据统计、进攻组织、队形空间以及分离的观察/解读/建议/限制。
- `tactical_review.md`：中文可阅读报告。
- `reviews/*.json`：各窗口模型审核与补证过程、响应 ID、实际送审帧路径。
- `evidence/`：完整上下文 MP4 和带时间戳图片。
- `api_usage.json`：实际请求次数、返回模型、token 用量及估算费用。
- `baseline_comparison.json`：与旧助手版本同 ID 节点的分类、球队与时间差，不是准确率报告。

成功窗口按源视频、报告、提示词、Schema、球队映射及模型签名缓存。
预算账本在重启后继续累计；失败请求保留保守预算占用，避免把超时当作免费。
费用按输入 $4/百万 token、输出 $20/百万 token 估算，未扣缓存优惠；是应用侧预估，不是平台硬限额。
实际账单以 OpenAI 控制台为准，上线前也应配置平台项目预算/限额。

## 验收

单元测试通过不代表足球识别准确。需要对照人工确认片段检查：事件精确率/召回率、时间定位、球队身份、门将误报和无证据评语。
旧助手版本并非人工真值，也可能被新版本纠正；不得仅以与旧版本一致来宣称同等或更高质量。
至少补充新比赛和负样本，统计不确定比例与事件覆盖率，避免通过全部拒绝来获得虚假的低误报。

## 本次五分钟验证

素材：`soccer_input_dataset/test_5min.mp4`，结果目录：`soccer_input_dataset/outputs/test_5min_openai_v1/`。

- 实际完成 58 次图像审核请求，估算费用上界 $8.0381；含开发期间的门将及边界复审回归。
- 本次调试累计预算上限调整为 $12；网页日常任务默认仍为 $8。
- 最终 25 个事件节点：采纳 21、待核实 2、排除 1、重复证据 1。
- 进攻组织 2 段，局部空间解读 8 段；没有发布精确阵型或空间距离。
- 两处射门候选改为传中；三个原始门将候选归类为脚下出球；约 107 秒等待重启不计为新出界。
- 80 项单元测试通过；桌面/手机视频播放、时间跳转、审核筛选和分析页签通过浏览器测试。
- 网页重新上传相同原视频已验证命中此 API 结果缓存；测试上传的临时副本已清理，没有重复调用模型。
- 不同新比赛的检测到 API 审核全链路已接入，但本轮没有另跑一场新比赛做端到端验收。

本轮复用已有检测框视频；真正重跑的是原始视频抽帧审核、补证与解读，不是重新检测跟踪。
当前仍有与旧助手版本不同的类别与时间，包括旧版本人工补入的部分节点未被新粗扫单独检出。
因此不能宣称已达到逐项审核的完全等效准确率；对照文件保留这些差异供进一步人工验收。

## 官方依据

- [GPT-5.6 Sol：图片输入、工具调用、结构化输出](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [图片输入](https://developers.openai.com/api/docs/guides/images-vision)
- [结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs)
