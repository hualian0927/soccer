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
python -m pip install -r requirements-analysis.txt
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

## 主要目录

```text
configs/                         模型与分析参数
tactical_analysis/               统一数据契约、19 个分析器、复核与证据工具
API/                             RF-DETR + SAHI + TCN 足球检测包（不是大模型接口）
web_preview/src/                 React 网页
web_preview/server/              上传、任务、媒体、人工审核、个人选段接口
run_local_video_gsr_visualization.py  视频到 GSR 的主入口
run_tactical_analysis.py         GSR 到技战术报告
run_openai_tactical_review.py    Responses 协议时序视觉复核
build_formation_evidence.py      阵型原图、框线图和短片
build_player_focus_video.py      可选球员编号轨迹
export_player_focus.py           选定人物的证据片段与分析
examples/                       无视频、无权重的合成轨迹 Demo
tests/                          Python 回归测试
docs/                           环境、功能、运行、API 与研究说明
```

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
