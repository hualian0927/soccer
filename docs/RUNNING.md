# 当前项目运行手册

先完成 [环境配置](ENVIRONMENT.md)。以下命令从仓库根目录运行，路径需替换成自己的授权素材；不要直接使用开发机绝对路径。

## 1. 无权重冒烟 Demo

```bash
python -m pip install -r requirements-analysis.txt
python examples/structured_demo.py
```

在 `soccer_input_dataset/outputs/synthetic_demo/` 生成合成 GSR 数据及统一报告。没有真实比赛图像，不是效果演示。适合验证数据契约、19 个模块和 JSON/Markdown/CSV 输出。

## 2. 视频到 GSR

```bash
conda activate sports
python run_local_video_gsr_visualization.py \
  --input-video soccer_input_dataset/input_match.mp4 \
  --video-name SNGS-999 --max-frames 0 \
  --work-root soccer_input_dataset/outputs/local-match/gsr_work \
  --output-video soccer_input_dataset/outputs/local-match/gsr_visualized.mp4 \
  --team0-colors blue --team1-colors white --referee-colors red \
  --goalkeeper-team0-colors yellowgreen --recall-optimized
```

`--max-frames 0` 才是完整视频；脚本默认仅 120 帧。初测可用 `--max-frames 900`。仅确认覆盖旧输出时加 `--overwrite`。其他球队衣服颜色需换先验。主脚本生成 GSR 的关键路径：

```text
soccer_input_dataset/outputs/local-match/gsr_work/SoccerNetGS/test/SNGS-999/SNGS-999.json
```

可选身份复核：

```bash
python refine_gsr_video_identities.py \
  --video soccer_input_dataset/input_match.mp4 \
  --json-path soccer_input_dataset/outputs/local-match/gsr_work/SoccerNetGS/test/SNGS-999/SNGS-999.json \
  --output-json soccer_input_dataset/outputs/local-match/identities.json \
  --team0-colors blue --team1-colors white --referee-colors red
```

## 3. 分析与显示

```bash
python run_tactical_analysis.py \
  --json-path soccer_input_dataset/outputs/local-match/identities.json \
  --input-video soccer_input_dataset/input_match.mp4 \
  --output-dir soccer_input_dataset/outputs/local-match/report

python make_tactical_visualization_video.py \
  --input-video soccer_input_dataset/input_match.mp4 \
  --json-path soccer_input_dataset/outputs/local-match/identities.json \
  --output-video soccer_input_dataset/outputs/local-match/boxes.mp4 --boxes-only

python build_layered_tactical_review.py \
  --video soccer_input_dataset/input_match.mp4 \
  --report-json soccer_input_dataset/outputs/local-match/report/tactical_analysis_report.json \
  --output-dir soccer_input_dataset/outputs/local-match/layered
```

最后一步默认只整理候选与证据，不调用模型。外接模型见 [API 配置](API_INTEGRATION.md)。

主要产物：`tactical_analysis_report.json/.md`、`event_timeline.csv`、`match_summary.json`、`highlight_manifest.csv`、`l1_event_timeline.json/.csv`、`l1_summary.json`、`set_piece_manifest.csv`、`layered_analysis.json`。

定位球单独切片：

```bash
python export_set_piece_clips.py \
  --input-video soccer_input_dataset/input_match.mp4 \
  --manifest soccer_input_dataset/outputs/local-match/report/set_piece_manifest.csv \
  --output-dir soccer_input_dataset/outputs/local-match/set_piece_clips
```

高光和解释成片：`export_tactical_highlights.py`、`make_tactical_report_video.py --interleave-chapters`。请用 `--help` 查看各脚本必选参数；图像/视频文件不提交 Git。

## 4. 编号与个人复盘

```bash
python build_player_focus_video.py \
  --video soccer_input_dataset/input_match.mp4 \
  --gsr-json soccer_input_dataset/outputs/local-match/identities.json \
  --output-dir soccer_input_dataset/outputs/local-match/player_focus

python export_player_focus.py \
  --tracking-json soccer_input_dataset/outputs/local-match/player_focus/player_tracks.json \
  --player-id 1 --start 10 --end 15 \
  --output-dir soccer_input_dataset/outputs/local-match/player_clip
```

编号 1 仅为格式示例，实际需选输出中可见的编号和时间段；不能跨切镜。`--render` 可给编号生成器额外导出整片编号视频；网页默认采用可关闭图层。

## 5. 阵型可视化

```bash
python build_formation_evidence.py \
  --video soccer_input_dataset/input_match.mp4 \
  --gsr-json soccer_input_dataset/outputs/local-match/identities.json \
  --bundle soccer_input_dataset/outputs/local-match/layered/layered_analysis.json \
  --tracking-json soccer_input_dataset/outputs/local-match/player_focus/player_tracks.json \
  --output-dir soccer_input_dataset/outputs/local-match/formation
```

输出代表帧原图、框线图、标注短片与增强分层报告。`--review-json` 是可选人工/助手复核输入，不要把 `review_annotations/test_5min_formation_frames.json` 用到别的视频。

## 6. 网页

```bash
conda activate sports
cd web_preview
npm ci
npm run dev -- --host 127.0.0.1 --port 4173
```

主流程：上传 → 检测/映射 → 身份复核 → 框选视频 → 19 个分析器 → 分层证据或外部模型 → H.264 转码 → 编号轨迹 → 阵型证据 → 项目分析页。

当前自动上传流程默认蓝队/白队/红衣裁判；其他配色需调整 `web_preview/server/local-analysis.js` 调用参数或先用命令行。代码里只有部分既有样例的固定缓存映射；新克隆不会自动带回开发机历史缓存。

本地任务在 `web_preview/runtime/`，日志见各任务 `analysis.log`。可读进度接口 `/api/analysis/:jobId`，事件修改接口 `PATCH /api/analysis/:jobId/events/:eventId`，个人片段接口 `POST /api/player-review`。本地 API 未做公网身份认证。

## 7. 其他入口

| 入口 | 用途 |
| --- | --- |
| `API/run.py --input VIDEO --output-dir DIR` | RF-DETR 检测 + TCN 足球补全 |
| `API/detect.py` / `API/complete.py` | 两阶段分别执行，完整参数见 API/README.md |
| `analyze_full_pitch_projection.py --csv ... --geometry ... --video ... --output ...` | 已有全场位置数据的空间分析 |
| `analyze_projection_folder.py --folder ... --output-dir ...` | 特定格式的双机位目录分析及报告 |
| `analyze_projection_with_ball.py` / `render_projection_l1_video.py` | 融合球轨迹与事件展示 |
| `make_individual_technique_analysis_video.py` | 近景姿态、射门与门将动作候选 |
| `run_match_analysis_workflow.py` | 长比赛扫描与远近景路由 |
| `run_fast_long_video_gsr.py` | 长视频替代快速流程，不等同于原基线所有步骤 |

这些输入契约不同，请先看脚本 `--help` 与相应 docs。API 目录是足球检测算法包，外接语言/视觉模型代码在根目录与 `tactical_analysis/openai_review.py`。

## 8. 验证与交付

```bash
python scripts/check_environment.py
python -m unittest discover -s tests -q
cd web_preview
node --test scripts/test-review-state.mjs
npm run build
```

Playwright 脚本 `scripts/verify-formation-focus.mjs`、`scripts/verify-player-focus.mjs` 使用本地五分钟素材及其生成目录，并依赖浏览器二进制；不是空仓库可直接通过的测试。`verify-player-focus.mjs` 支持 `PLAYWRIGHT_CHROMIUM_PATH`，部分其他脚本仍需按本机修改路径。测试服务默认 `http://127.0.0.1:4173`，可通过 `PREVIEW_URL` 指定。

换机器时无需复制 Conda 环境、node_modules 或 runtime；按环境文件重装，单独取得权重与授权素材，再运行。迁移旧报告中的绝对路径需要检查和重新生成，不能仅复制 JSON 就认为视频可播放。
