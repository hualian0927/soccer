# 赛析本地网页预览

该目录是足球 L1 事件分析的本地前端，复用现有 Python 推理和视频生成流程。

## 本地运行

```bash
cd /home/lhl/ai_models/sports/web_preview
npm install
npm run dev
```

默认地址为 `http://localhost:4173/`。如果端口已占用，Vite 会选择下一个可用端口。

## 已实现交互

- 拖拽或选择本地视频，生成预览与缩略图并提交本机分析服务。
- 将视频添加至现有项目，或新建项目并添加。
- 在项目列表中查看项目，再进入项目视频列表。
- 点击项目视频进入 L1 事件分析工作台。
- 点击进度条节点或右侧事件，同步跳转播放器时间。
- 按射门、定位球、传接带、球权转换、防守干预和门将事件筛选。
- 对当前事件执行确认或拒绝，结果写入本地 `review_log.json`。
- 定位球页先统计角球、任意球、球门球、界外球、开球和点球，再逐个播放独立片段。
- 响应式桌面与移动端布局。

## 本地分析流水线

上传视频后，Vite 本地服务会把文件保存到忽略提交的 `web_preview/runtime/`，随后依次执行：

```text
run_local_video_gsr_visualization.py
  -> make_tactical_visualization_video.py
  -> run_tactical_analysis.py
  -> export_set_piece_clips.py
  -> FFmpeg H.264 网页兼容转码
```

网页工作台只呈现 L1 事件，不再展示原来的阵型、压迫、多打少和防守空档等 P1-P3 战术结论。`test_5min.mp4` 会通过 SHA-256 匹配并复用同源 GSR 分析；其他视频会在 `sports` Conda 环境中执行完整流程。

当前默认球衣先验与此前样例一致：蓝队、白队、红衣裁判、黄绿色门将归入蓝队。完整新视频处理时间取决于时长、GPU 和模型状态。

本地服务使用以下输出：

- `tactical_analysis_report.json`
- `l1_event_timeline.json`
- `l1_event_timeline.csv`
- `l1_summary.json`
- `set_piece_manifest.csv`
- `set_piece_clips/set_piece_clips.json`

## 本地接口

- `POST /api/analysis`：上传视频并启动完整本地分析。
- `GET /api/analysis/:jobId`：查询处理进度和结果。
- `GET /api/analysis/:jobId/l1`：获取 L1 摘要、事件和定位球片段。
- `PATCH /api/analysis/:jobId/events/:eventId`：确认或拒绝事件候选。

## 构建

```bash
npm run build
```

生产构建输出在 `dist/`，该目录默认不提交。
