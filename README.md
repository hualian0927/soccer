# 赛析：本地足球视频技战术分析

基于 SoccerNetGSR / Broadcast2Pitch 的视觉分析扩展，包含检测跟踪、球场投影、事件候选、技战术报告、可选视觉模型复核和 React 本地网页 Demo。仅使用视频与视觉轨迹，不依赖 GPS 或可穿戴设备。

当前交付分支：`code`。运行说明更新于 2026-10-09。

> 本项目是候选分析与复盘工具，不是官方比赛数据供应商。功能可运行不代表已通过跨比赛准确率验证；二维投影、抽帧审核与几何规则均存在误差，需保留视频证据供复核。

## 文档导航

| 内容 | 文档 |
| --- | --- |
| Python/CUDA/Node、权重、配置与排错 | [环境配置](docs/ENVIRONMENT.md) |
| 全部功能、实现状态、技术路线与边界 | [功能清单](docs/FEATURES.md) |
| 命令行、网页、个人选段、阵型与双机位 | [运行手册](docs/RUNNING.md) |
| 外接模型协议、密钥、调用入口及费用约束 | [API 配置](docs/API_INTEGRATION.md) |
| 前端与本地服务 | [Web Demo](web_preview/README.md) |
| 原实现与研究出处 | [上游 README](docs/UPSTREAM_README.md)、[论文技术借鉴](docs/论文技术借鉴与项目落地总结.md) |

## 处理流程

```text
原始比赛视频
  -> 人/球检测、跟踪、球队/裁判/门将区分、球场标定
  -> 统一 GSR 时序数据
  -> 19 个分析器：事件、空间、射门、组织、防守、转换等候选
  -> 前后视频证据、可选视觉模型复核、结构化评语
  -> 中文报告、事件索引、定位球切片、网页播放与审核
  -> 按需启用：阵型框线图/短片、球员编号及个人片段分析
```

二维轨迹负责位置与时序候选；原图负责接触、重启动作、门将身份等证据；审核结果与算法推断分开保存。没有 API 时不会伪造模型已审核结果。

## 快速开始

```bash
git clone --branch code https://github.com/hualian0927/soccer.git
cd soccer
conda create -n sports python=3.10 ffmpeg=4.4 -c conda-forge -y
conda activate sports
python -m pip install -r requirements/analysis.txt
python examples/structured_demo.py
```

以上为无需权重、无需 API 的合成轨迹冒烟 Demo，输出 JSON/Markdown/CSV 报告，但不代表真实比赛效果。完整视频推理还需按[环境配置](docs/ENVIRONMENT.md)安装 PyTorch、完整依赖与权重。

启动网页：

```bash
cd web_preview
npm ci
npm run dev -- --host 127.0.0.1 --port 4173
```

访问 `http://localhost:4173`，使用“素材库 → 上传 → 添加项目 → 打开项目视频”。处理会调用同一 WSL/Linux 中的 `sports` Conda 环境，不会自动调用其他机器或 Windows Python。

**首次克隆不包含历史比赛视频与输出。** 历史项目入口引用开发机缓存，缺少文件时不能播放，并非代码缺失。请上传自己的授权素材并完成处理。只安装 Node 可启动界面；没有模型与 Python 环境时不能完成视频分析。`npm run build` 只构建静态前端，本地 API 由 `npm run dev` 加载，不能仅用静态托管代替后端。

## 目录与文件用途

根目录仅保留 `README.md`（说明）、`setup.py`（YOLOX 安装入口）、`requirements.txt`（完整 Python 依赖）、`.env.example`（空密钥配置模板）和 `.gitignore`（排除素材及敏感文件）。原根目录 41 个 Python 文件已按功能移到 `workflows/`，不保留重复旧入口。

### 工作流脚本

以下文件均位于对应目录。`__init__.py` 用于声明 Python 包，不会自动执行分析或调用模型。

| 目录 | 文件与作用 |
| --- | --- |
| `workflows/gsr/` 检测与比赛状态 | `run_local_video_gsr_visualization.py`：视频到 GSR 的完整管线；`run_fast_long_video_gsr.py`：长视频轻量预览；`inference_soccernetGSR.py`：检测、跟踪及角色识别；`kpts.py`：球场关键点与投影；`ball_detection_adapter.py`：足球检测结果适配与轨迹合并；`write_json_file_team.py`：导出统一比赛状态 JSON。 |
| `workflows/identity/` 球队身份 | `jersey_color.py`：球衣颜色特征；`team_identity.py`：轨迹身份推断；`tracklet_appearance.py`：CLIP 外观复核；`refine_team_identities.py`：跟踪文件的球队修正；`refine_gsr_video_identities.py`：结合视频修正 GSR 身份；`refine_referee_roles.py`：裁判角色修正及重绘。 |
| `workflows/tactical/` 分析入口 | `run_tactical_analysis.py`：调用统一分析器输出报告；`analyze_tactical_metrics.py`：早期基础指标统计；`run_match_analysis_workflow.py`：长比赛镜头扫描、远近景分流与分段处理。 |
| `workflows/projection/` 全景与双机位 | `analyze_full_pitch_projection.py`：全场位置与空间统计；`analyze_projection_with_ball.py`：加入足球、边界及重启候选；`analyze_projection_folder.py`：成组处理素材并生成报告。 |
| `workflows/visualization/` 视频与切片 | `make_tactical_visualization_video.py`：检测框、战术板或仅框视频；`make_tactical_report_video.py`：插入报告结论；`make_formation_analysis_video.py`：阵型视频；`build_formation_evidence.py`：阵型原图、框线图及短片；`make_offensive_analysis_video.py`：进攻可视化；`make_individual_technique_analysis_video.py`：射门/门将姿态窗口；`make_vision_review_video.py`：视觉复核结果展示；`render_projection_l1_video.py`：二维事件视频；`visualize_prediction_results.py`：原始预测绘制；`visualize_local_result.py`：本地结果绘制入口；`visualize_referee_distinct_colors.py`：裁判独立配色；`export_tactical_highlights.py`：高光裁剪；`export_set_piece_clips.py`：定位球分类切片。 |
| `workflows/review/` 证据与模型审核 | `run_openai_tactical_review.py`：Responses API 时序复核；`review_tactical_candidates_with_vision.py`：候选视觉复核；`build_layered_tactical_review.py`：生成分层分析包及可选 API 评语；`build_assistant_review_version.py`：合并已有审核标注；`build_reviewed_tactical_layers.py`：将已审核评语补入层级；`prepare_assistant_review_sheets.py`：抽帧审核拼图；`prepare_layered_review_evidence.py`：分层节点证据。 |
| `workflows/players/` 个人选段 | `build_player_focus_video.py`：生成可开关的球员编号和轨迹；`export_player_focus.py`：指定球员时间段的短片与报告。 |
| `workflows/data/` 资源下载 | `download_properties.py`：按配置下载模型资源，下载权重不提交。 |

### 分析库与网页

| 目录 | 文件与作用 |
| --- | --- |
| `tactical_analysis/` | `models.py`、`base.py`：数据契约和分析器接口；`gsr_io.py`：读取 GSR；`features.py`：时序特征；`pipeline.py`：调度；`reporting.py`：报告、高光和摘要；`l1.py`、`projection_l1.py`：事件转换；`openai_review.py`：API 请求、预算与缓存；`reviewed_layers.py`：复核结果分层；`player_tracker.py`、`player_focus.py`：编号跟踪和个人片段；`formation_evidence.py`：阵型证据几何。 |
| `tactical_analysis/analyzers/` | `quality.py`：质量门控；`events.py`：基础时序事件；`ball_out.py`：出界；`spatial.py`：空间统计；`formation.py`：阵型；`team_shape.py`：宽度/纵深/线距；`set_pieces.py`：定位球；`goalkeeper.py`：门将候选；`progression.py`：推进；`attacking_chains.py`：进攻链；`defensive_risk.py`：防守风险；`defensive_interventions.py`：防守干预；`defensive_gaps.py`：空档；`relations.py`：局部关系；`numerical_superiority.py`：人数优势；`pressing.py`：压迫；`transitions.py`：转换；`pass_network.py`：传球网络；`phases.py`：比赛阶段。 |
| `web_preview/src/` | `main.jsx`：挂载入口；`App.jsx`：素材、项目、分析和播放页面；`data.js`：样例项目元数据；`LayeredAnalysisPanel.jsx`：分层评语；`SpatialAnalysisPage.jsx`：空间页；`SwitchableCameraStage.jsx`：双机位切换；`PlayerFocus.jsx`：人物选段；`FormationEvidence.jsx`：阵型证据；`review-state.js`：复核状态；各 `.css`：对应界面样式。 |
| `web_preview/server/` | `local-analysis.js`：上传、分析队列、报告和媒体接口；`player-focus.js`：个人选段导出接口。 |
| `web_preview/` | `package.json`、`package-lock.json`：前端依赖；`vite.config.js`：开发服务与本地接口挂载；`index.html`：网页入口；`scripts_extract_posters.py`：封面提取；`README.md`：网页运行说明。 |
| `web_preview/scripts/` | `test-review-state.mjs`：状态单测；`verify-preview.mjs`：页面回归；`verify-player-focus.mjs`：个人选段回归；`verify-formation-focus.mjs`：阵型交互回归；`verify-openai.mjs`：审核结果页面回归。浏览器回归需要指定历史素材，不会自动进行付费审核。 |

### 配置、工具与上游组件

| 目录 | 文件与作用 |
| --- | --- |
| `configs/` | `config.yaml`：检测、跟踪、权重及路径；`tactical_analysis.yaml`：分析器开关和阈值。 |
| `requirements/` | `analysis.txt`：仅结构化分析的轻量依赖；`llama.txt`：可选 LLaMA 球衣识别依赖。 |
| `scripts/` | `check_environment.py`：只读环境诊断；`audit_release.py`：暂存区密钥/大文件检查；`refine_tracklets.sh`：轨迹后处理批处理。 |
| `examples/` | `structured_demo.py`：无权重、无视频、无 API 的合成 GSR Demo。 |
| `tests/` | `test_*.py`：按文件名对应功能的回归测试，包括颜色身份、事件、门将、阵型、投影、模型审核、个人选段及目录迁移。 |
| `review_annotations/` | `test_5min_assistant_v3.json`：事件复核；`test_5min_layered_v4.json`：分层评语；`test_5min_formation_frames.json`：阵型证据帧。仅属于开发样例。 |
| `docs/` | `ENVIRONMENT.md`：环境；`RUNNING.md`：运行；`FEATURES.md`：功能边界；`API_INTEGRATION.md`：外接模型；`FILE_INDEX.md`：逐文件用途索引；其他 Markdown：研究、进度和历史设计；`render_tactical_project_flowchart.py`：流程图生成。 |
| `API/` | `run.py`：足球检测完整入口；`detect.py`：检测；`complete.py`：轨迹补全；`requirement.txt`：该子包依赖；`football_inference/api.py`：调用接口；`pipeline.py`：RF-DETR/SAHI/TCN 管线；`tcn_model.py`：时序补全网络。不是外接语言模型接口。 |
| `IDATR/` | `Tracklet.py`：轨迹对象；`rmv_doub_bbox.py`：去重；`gen_tracklets.py`：轨迹段生成；`refine_tracklets.py`：轨迹修正；`create_court_file.py`：球场坐标文件。 |
| `jersey_model/` | `CLIPFinetune.py`：球衣/角色 CLIP 网络。 |
| `exp/` | `yolox_x_soccernet.py`：SoccerNet YOLOX 实验配置。 |
| `sfr/` | `inference.py`：球场分割推理；`models/`：分割及注意力网络；`utils_opt.py`：标定优化；`template/`：球场几何模板。 |
| `template/`、`assets/fonts/` | `soccernet_template_97.npy`：球场模板；`NotoSansCJKsc-Regular.otf`：中文绘制字体。 |
| `utils/` | `bbox.py`：框几何；`sys_utils.py`：张量/系统辅助；`transforms.py`：图像变换。 |
| `yolox/` | 上游检测框架；`models/`：网络；`data/`：预处理和数据加载源码；`exp/`：配置基类；`EIoU_tracker/`、`byte_tracker/`：跟踪；`evaluators/`：评测；`layers/`：算子；`tracking_utils/`、`utils/`：工具。 |
| `sn-trackeval/` | 上游跟踪评测；`trackeval/datasets/`：数据适配；`metrics/`：HOTA/ID 等指标；`scripts/`：各数据集评测入口；`tests/`：上游测试；`docs/`：格式与评测说明；`LICENSE`：许可。 |

全部已提交文件的路径与简述见 [逐文件索引](docs/FILE_INDEX.md)。保留上游组件内部目录，避免破坏它们的安装和导入约定。

### 本地资源目录（不上传）

`checkpoints/` 保存模型（包括迁入的 `yolov8n.pt`、`yolov8n-pose.pt`）；`data/archives/` 保存原根目录 ZIP；`docs/local_research/` 保存个人调研 Markdown/PDF；`data/`、`soccer_input_dataset/`、`二维分析/`、`Sportec Open DFL/` 保存素材与结果；`sports-paper/` 保存论文；`web_preview/runtime/` 保存上传任务。已存在的素材目录不再移动，避免历史项目媒体路径失效。缓存不属于运行源码，不纳入 Git。

## 目录迁移后的调用方式

在**仓库根目录**运行，使用模块入口，替代原来的 `python 文件名.py`；不直接执行 `python workflows/.../文件名.py`。网页后端及工作流子进程已同步更新，不需要手工更改前端。

```bash
conda activate sports
python -m workflows.gsr.run_local_video_gsr_visualization --help
python -m workflows.tactical.run_tactical_analysis --help
python -m workflows.review.run_openai_tactical_review --help
python -m workflows.players.export_player_focus --help
```

业务参数不变，完整命令见 [运行手册](docs/RUNNING.md)。旧自动化任务若仍引用根目录脚本，需按上表改成 `python -m workflows.分类.脚本名`。

## 验证

```bash
conda activate sports
python scripts/check_environment.py
python -m unittest discover -s tests -q
cd web_preview
node --test scripts/test-review-state.mjs
npm run build
```

Playwright 比赛播放测试需要另行准备对应视频和分析结果，详见运行手册。测试通过不代表赛事识别精度已量化。

## 不上传的内容

模型权重、比赛视频、图片抽帧、数据集、生成报告与缓存、`node_modules`、构建结果、真实 `.env` 和 API 密钥均不提交。`review_annotations/` 仅保存开发样例复核记录，不含视频，不会自动适用于其他比赛。

密钥仅放后端环境变量，示例见 [.env.example](.env.example)。本地 API 无公网用户认证，应仅监听本机。对外部署需补充认证、配额、任务隔离与文件权限。只有获得相应授权的比赛视频才可提交外部模型。

## 上游与许可

本地扩展源于 [yinmayoo185/SoccernetGSR](https://github.com/yinmayoo185/SoccernetGSR)，使用仓库现有 YOLOX、Torchreid、CLIP、SoccerNet 评测等组件。保留源文件版权信息与现有许可证；第三方代码、模型、字体、数据授权分别适用，此分支不代表对所有内容统一授予商业许可。SoccerNet 赛事素材需自行取得授权，不随代码分发。
