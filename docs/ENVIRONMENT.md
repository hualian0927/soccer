# 当前项目运行环境与配置

更新：2026-10-09。此文件是 `code` 分支的环境配置入口；旧配置文章用于历史参考。

## 1. 分层安装

| 用途 | 需要 | 不需要 |
| --- | --- | --- |
| 网页 UI 开发 | Node.js、npm、`web_preview/package-lock.json` | Python、权重、API 密钥 |
| 已有 GSR JSON / 合成 Demo 分析 | Python 3.10、`requirements-analysis.txt` | GPU、深度模型、API |
| 完整视频检测与网页上传处理 | `sports` Conda、FFmpeg、PyTorch、`requirements.txt`、Torchreid、YOLOX 扩展及权重 | 外部模型 API 可选 |
| 个人动作与临时编号 | 完整环境、Ultralytics、检测/姿态权重 | 必须拥有 API 不是前提 |
| RF-DETR + TCN 足球检测 | `API/requirement.txt`、其两份专用权重 | 这不是聊天大模型 API |
| 外部视觉复核 | 图像抽帧环境、jsonschema、网络、支持图片的模型权限 | 不需要额外安装 OpenAI SDK，本项目用标准库 HTTP |
| LLaMA 球衣识别 | 额外 `requirements-llama.txt` 与相关权重 | 默认 CLIP 路径不需要 |

## 2. 开发机观察值与兼容边界

本机为 Ubuntu / WSL2，Python 3.10，Node 22.22.1、npm 10.9.4。已安装包观察值：

| 包 | 版本 |
| --- | --- |
| torch / torchvision / torchaudio | 2.5.1 / 0.20.1 / 2.5.1 |
| numpy / pandas / scipy | 2.2.6 / 2.3.3 / 1.15.3 |
| ultralytics / SoccerNet | 8.4.37 / 0.1.62 |
| rfdetr / sahi | 1.9.2 / 0.11.36 |
| timm / torchreid / jsonschema | 1.0.28 / 1.4.0 / 4.26.0 |
| transformers | 5.14.1 |

这是开发机快照，不是已验证的全新安装锁文件。本机同时装有多个 OpenCV distribution，不建议照搬；新环境只保留所需的一种 cv2 提供包，主流程需 contrib 中的光流相关接口。`requirements.txt` 部分依赖仍是开放版本，安装后必须自检，完整 CUDA 环境尚未做独立机器重装验收。

## 3. 系统与 Python

在仓库根目录执行，不要修改系统 Python：

```bash
sudo apt update
sudo apt install -y git build-essential libglib2.0-0 libgl1
conda create -n sports python=3.10 ffmpeg=4.4 -c conda-forge -y
conda activate sports
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy cython
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/KaiyangZhou/deep-person-reid.git --no-build-isolation
python -m pip install -e . --no-build-isolation
python scripts/check_environment.py
python -m pip check
```

上面的 PyTorch 是项目历史 CUDA 12.1 组合；GPU 驱动需与之兼容。WSL 使用 Windows 提供的 NVIDIA 驱动支持，先检查 `nvidia-smi`。CPU 能运行轨迹分析；完整检测/近景姿态不保证每条脚本均支持纯 CPU。

`requirements.txt` 已包含 Ultralytics、SoccerNet、RF-DETR、SAHI、jsonschema 等本地扩展依赖。LLaMA 是可选且可能改变 torch 依赖组合，应在单独环境测试后再启用，不要为默认 CLIP 模式强装 Unsloth。

若只测试结构化分析：

```bash
python -m pip install -r requirements-analysis.txt
python examples/structured_demo.py
```

## 4. 模型与字体

代码不包含权重。`python download_properties.py` 可尝试获取原仓库权重，下载链接、访问权限与上游可用性由提供方决定。请核对实际落盘文件：

| 文件/资源 | 用途 |
| --- | --- |
| `checkpoints/yolox_soccernet.pth.tar` | 基线人/球检测 |
| `checkpoints/sports_model.pth.tar-60` | OSNet ReID |
| `checkpoints/SoccernetGSR_EfficientNet_Best.pth` | 场地关键点与映射 |
| `checkpoints/CLIP_Jersey.pth` | CLIP 球衣相关识别 |
| `checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth` | 原配置可能用到的 OSNet 预训练资源 |
| `yolov8n.pt`、`yolov8n-pose.pt` 等脚本指定文件 | 补检、临时编号与姿态分支 |
| `API/model/rfdetr_ball_best_ema.pth` | RF-DETR 足球检测 |
| `API/model/tcn_ball_completion_best.pt` | TCN 补全轨迹 |
| `assets/fonts/NotoSansCJKsc-Regular.otf` | 中文视频/图像字体，仓库已有资源 |

部分框架会首次联网下载预训练骨干或 CLIP 权重；离线运行须预先准备。权重来源及用途授权需单独确认。

主配置见 `configs/config.yaml`，分析参数见 `configs/tactical_analysis.yaml`。不要只拷贝 Python 文件而遗漏配置、字体和前端 lockfile。

## 5. 网页环境

```bash
conda activate sports
cd web_preview
npm ci
npm run dev -- --host 127.0.0.1 --port 4173
```

浏览器访问 `http://localhost:4173`。若端口占用，查看终端实际地址。服务端进程必须能找到 `conda`；个人片段导出默认也使用 `conda run -n sports python`，或以 `SPORTS_PYTHON` 指定解释器。`SPORTS_PYTHON` 目前不替换整条上传链的 Conda 调用。

网页代码、上传 API 和媒体 Range 服务在同一个 Vite 开发服务器运行。`npm run preview` 或直接打开 `dist/index.html` 不会启动 Python API。

## 6. 外接模型环境

复制根目录 `.env.example` 为本机 `.env` 后填写，不提交：

```bash
cp .env.example .env
# 编辑 .env；禁止使用 VITE_ 前缀保存密钥。
set -a
source .env
set +a
```

Python 与现有服务读取进程环境，不自动读取此文件。需在同一终端启动网页。空密钥代表不启用外部调用；不要依赖模型名推断它一定支持图片。详细参数、接口兼容限制、费用风险见 [API 配置](API_INTEGRATION.md)。

## 7. 常见故障

交付验证记录（2026-10-09）：从 Git 暂存区导出干净代码快照，在独立 Python 3.10 venv 中仅安装 `requirements-analysis.txt`，19 个分析器的合成 Demo 成功。干净快照使用既有 sports 依赖运行 98 项测试全部通过；前端 2 项逻辑测试、构建及桌面/手机个人片段导出回归通过。完整 GPU 推理环境未在全新机器重装，未做付费 API 在线验证。

- `conda: not found`：从初始化过 Conda 的终端启动网页，确认 `conda run -n sports python --version`。
- `torchreid`/`yolox._C` 不可导入：先装 PyTorch、编译工具和 Cython，再执行上述源安装命令。
- `cv2` 冲突：检查 `pip list` 是否混装多个 OpenCV 包，统一后再测 contrib 功能；不要盲目复制开发机混装快照。
- 历史项目没有封面或播放失败：视频与输出未进 Git。使用自己的授权素材重新处理。
- 视频生成却不能播放：检查 FFmpeg 有 H.264 编码器，网页需要 H.264 + yuv420p；服务支持 Range。
- AI 调用失败：查看节点状态与 `api_usage.json`，权限、图片、JSON schema、SSE 支持均须匹配；失败节点不得当作已支持。
- 内存或显存不足：降低检测 batch、先处理 30 秒，再放大；RF-DETR 切片检测与长片抽帧会增加磁盘占用。
- Playwright 缺系统库：在允许安装系统依赖的环境执行 `npx playwright install --with-deps chromium`；现有视觉回归脚本部分仍引用开发机浏览器路径，应按运行手册设置或调整。
