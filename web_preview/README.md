# 赛析本地网页预览

> 当前交付说明：以根目录 [README](../README.md)、[环境配置](../docs/ENVIRONMENT.md)、[运行手册](../docs/RUNNING.md) 和 [API 接入](../docs/API_INTEGRATION.md) 为准。代码包含全部 demo 组件，但不含历史比赛素材、缓存和密钥。下文部分固定样例描述为开发机历史运行记录，不能代替新环境的模型能力验证。

该目录是足球事件与分层分析的本地前端，复用现有 Python 推理和视频生成流程。

## 本地运行

```bash
cd web_preview
npm ci
npm run dev -- --host 127.0.0.1 --port 4173
```

默认地址为 `http://localhost:4173/`。如果端口已占用，Vite 会选择下一个可用端口。

## 已实现交互

- 拖拽或选择本地视频，生成预览与缩略图并提交本机分析服务。
- 将视频添加至现有项目，或新建项目并添加。
- 在项目列表中查看项目，再进入项目视频列表。
- 点击项目视频进入对应的 L1 事件或二维空间分析工作台。
- 点击进度条节点或右侧事件，同步跳转播放器时间。
- 按射门、定位球、传接带、球权转换、防守干预和门将事件筛选。
- 对当前事件执行确认或拒绝，结果写入本地 `review_log.json`。
- 定位球页先统计角球、任意球、球门球、界外球、开球和点球，再逐个播放独立片段。
- 响应式桌面与移动端布局。

## 本地分析流水线

上传视频后，Vite 本地服务会把文件保存到忽略提交的 `web_preview/runtime/`，随后依次执行：

```text
run_local_video_gsr_visualization.py
  -> refine_gsr_video_identities.py
  -> make_tactical_visualization_video.py --boxes-only
  -> run_tactical_analysis.py
  -> build_layered_tactical_review.py [--review-api]
  -> FFmpeg H.264 网页兼容转码
  -> build_player_focus_video.py
  -> build_formation_evidence.py
```

新上传视频的主画面只保留原片与人物、足球框；右侧按 L1 事件、L2 统计、L3 组织、L4 空间分层。模型文字及审核证据位于播放器下方，不烧录到画面。`test_5min.mp4` 会通过 SHA-256 匹配新版 `test_5min_layered_v2/` 缓存；其他视频会在 `sports` Conda 环境中执行完整流程。旧历史项目保留旧版展示。

启动 Vite 前可设置 `DEEPSEEK_API_KEY` 环境变量以启用真实视觉审核。不要使用 `VITE_` 前缀，不要把密钥写入前端或提交 Git。未配置时仍生成候选、证据图片与分层报告，明确标记尚未视觉复核，不模拟 API 结果。每片默认最多抽样审核 20 个节点，每个节点前后各 5 秒、41 张全景图；门将候选另附局部图并独立复审，调用可能产生费用。

`二维分析/` 中的 60 秒双机位俯视样例是单独的空间分析项目，不经过上传视频的 L1 流水线。它复用 `analyze_full_pitch_projection.py` 生成的 `spatial_analysis_report.json`，在网页中显示可见球员队形量化和逐时段视觉解释。该素材缺少足球轨迹与原始比赛画面，因此不判定传球、射门、角球或控球。详情见 `二维分析/空间技战术分析说明.md`。

素材库和“五分钟比赛技战术复盘”项目中另有“五分钟定位球候选复核（修正版）”。它直接播放完整的视觉复核成片，不进入旧 L1 工作台；54 秒与 98 秒的角球几何候选均未获视觉确认。独立证据片段和模型复核 JSON 位于 `soccer_input_dataset/outputs/test_5min_corner_review_v2/`，目录内的 `复核说明.md` 记录了切片修正和证据边界。
上述定位球修正版保留为历史结果；当前重新上传同一 `test_5min.mp4` 命中的是分层新版，不再使用该旧缓存。

当前默认球衣先验沿用样例：蓝队、白队、红衣裁判，身份复核将黄/绿色作为蓝队门将候选。此先验并非跨比赛通用，其他配色需要调整服务端参数或单独运行脚本；门将角色和归队仍可能误判。完整新视频处理时间取决于时长、GPU 和模型状态。

本地服务使用以下输出：

- `tactical_analysis_report.json`
- `l1_event_timeline.json`
- `l1_event_timeline.csv`
- `l1_summary.json`
- `set_piece_manifest.csv`
- `set_piece_clips/set_piece_clips.json`

新版另输出 `layered_analysis.json`、`reviews/*.deepseek.json` 和 `evidence/<event_id>/` 中的 10 秒原始证据片段、逐帧图片。分层新版入口为项目列表中的“五分钟分层技战术复盘”。详细边界见 [分层视觉复核与门将改进](../docs/分层视觉复核与门将改进.md)。

## 本地接口

新增 OpenAI 自动视觉审核路径：启动后端前设置 `OPENAI_API_KEY`，新上传视频会调用
`run_openai_tactical_review.py`，固定使用 `gpt-5.6-sol`，优先于旧 DeepSeek 路径。
该路径包含动态补帧、出界去重复审、门将独立复审和分层评语，默认每任务最多 80 次请求、8 美元预估预算。
密钥仅用于后端，不应使用 `VITE_` 前缀。终端交互输入只在当前进程中有效，重启服务需重新设置。
完整配置、费用与准确性边界见 [OpenAI 视觉审核工作流](../docs/OpenAI视觉审核工作流.md)。
新样例项目名称为“五分钟智能视觉复盘”，旧版本保留用于对照。

- `POST /api/analysis`：上传视频并启动完整本地分析。
- `GET /api/analysis/:jobId`：查询处理进度和结果。
- `GET /api/analysis/:jobId/l1`：获取 L1 摘要、事件和定位球片段。
- `PATCH /api/analysis/:jobId/events/:eventId`：确认或拒绝事件候选。
- `GET /api/media?path=...`：读取项目内视频和报告；视频支持 HTTP Range，可播放和拖动进度条。

## 构建

```bash
npm run build
```

生产构建输出在 `dist/`，该目录默认不提交。
