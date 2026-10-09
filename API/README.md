# RF-DETR + TCN 足球推理包

> Git 代码分支不包含 `model/` 下的实际权重与示例视频；需从有权分发的提供方另行取得。此目录是本地足球检测算法包，不是外接大模型 API。

该目录是可独立复制和交付的足球视频推理包。检测阶段逐帧执行 RF-DETR + Global SAHI，再由双向 TCN 读取已保存的检测结果并补全漏检帧的足球轨迹。两个阶段可以分别运行，也可以一键串行运行。

## 目录结构

```text
API/
├── football_inference/
│   ├── __init__.py
│   ├── api.py
│   ├── pipeline.py
│   └── tcn_model.py
├── model/
│   ├── rfdetr_ball_best_ema.pth
│   ├── tcn_ball_completion_best.pt
│   └── SHA256SUMS
├── README.md
├── complete.py
├── detect.py
├── requirement.txt
└── run.py
```

所有内置资源均按本文件所在目录定位。默认模型路径为 `model/rfdetr_ball_best_ema.pth` 和 `model/tcn_ball_completion_best.pt`，移动整个 `API/` 目录后无需修改代码。

## 环境要求

- Python 3.10
- 建议使用 NVIDIA GPU；CPU 可以运行，但 RF-DETR 视频推理会很慢
- 可选安装带 `libx264` 和 `aac` 编码器的 FFmpeg。可用时输出为 H.264 并保留原视频音频；不可用时自动输出 OpenCV 生成的无声 MP4

在本目录中创建环境并安装依赖：

```bash
cd API
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

如需指定 CUDA 版本，请先按照 PyTorch 官方说明安装与机器 CUDA 匹配的 `torch==2.5.1` 和 `torchvision==0.20.1`，再执行依赖安装命令。

## 分两步调用

以下命令均在 `API/` 目录中执行。

第一步，使用 RF-DETR + Global SAHI 检测：

```bash
python detect.py \
  --input example/input.mp4 \
  --output-dir output/detection
```

该步骤生成：

| 文件 | 内容 |
| --- | --- |
| `detection.mp4` | 绘制全部 NMS 后检测框的视频 |
| `detections.json` | 视频元数据、SAHI 参数和每帧全部候选框，供 TCN 阶段读取 |
| `detections.csv` | 检测结果表格，每个候选框一行 |

第二步，读取第一步结果并运行 TCN：

```bash
python complete.py \
  --input example/input.mp4 \
  --detections output/detection/detections.json \
  --output-dir output/completion
```

第二步必须使用第一步对应的原始视频。程序会校验视频宽高、帧率和帧数，防止检测结果与视频错配。该阶段只加载 `model/tcn_ball_completion_best.pt`，不会重新运行 RF-DETR。

Global SAHI 检测的常用可选参数：

```text
--device auto|cuda|cpu   推理设备，默认 auto
--confidence 0.18        RF-DETR 置信度阈值
--tile-size 672          SAHI 切片宽高
--batch-size 4           每次送入 RF-DETR 的切片数
--overlap 0.2            相邻切片重叠比例
--nms-iou 0.5            全图候选框 NMS IoU 阈值
--weights PATH           替换 RF-DETR 权重
```

TCN 补全支持 `--device` 和 `--weights`。命令中的输入、输出和替换权重均可使用相对路径。

## 一键调用

`run.py` 会依次完成上述两个阶段，并将所有结果写入同一个目录：

```bash
python run.py \
  --input example/input.mp4 \
  --output-dir output
```

默认自动选择 CUDA 或 CPU，并自动加载 `model/` 中的两个权重。可选参数与两个分步脚本一致，其中 `--resolution` 对应 `detect.py` 的 `--tile-size`：

```text
--device auto|cuda|cpu   推理设备，默认 auto
--confidence 0.18        RF-DETR 置信度阈值
--resolution 672         RF-DETR 输入分辨率
--batch-size 4           SAHI 切片批量大小
--overlap 0.2            SAHI 切片重叠比例
--nms-iou 0.5            全图 NMS IoU 阈值
--rfdetr-weights PATH    替换 RF-DETR 权重
--tcn-weights PATH       替换 TCN 权重
```

命令中的输入和输出均可使用相对路径。程序运行期间会在标准输出打印 `PROGRESS` 开头的 JSON 进度信息。

## Python 分步调用

```python
from football_inference import complete_video_trajectory, detect_video_sahi


detections_path = detect_video_sahi(
    input_path="example/input.mp4",
    output_dir="output/detection",
    device="auto",
    confidence=0.18,
    tile_size=672,
)

complete_video_trajectory(
    input_path="example/input.mp4",
    detections_path=detections_path,
    output_dir="output/completion",
    device="auto",
)
```

一键 Python 接口为 `infer_video(...)`。三个公开接口都默认根据推理包自身位置加载 `model/` 中的权重，因此从其他工作目录调用也不需要填写模型路径。输入、输出以及可选的替换权重路径可传入字符串或 `pathlib.Path`。

## 输出文件

TCN 阶段会在 `--output-dir` 下生成：

| 文件 | 内容 |
| --- | --- |
| `detection.mp4` | RF-DETR 检测框视频 |
| `tcn_completed.mp4` | 检测点与 TCN 补全轨迹视频 |
| `trajectory.jsonl` | 每帧检测、TCN 预测及最终坐标 |
| `trajectory.csv` | 便于表格软件处理的轨迹结果 |
| `summary.json` | 视频信息、状态计数和推理参数 |

`output_status` 的取值为：`detected` 表示 RF-DETR 直接检测，`completed_missing` 表示漏检帧由 TCN 补全，`invalid` 表示没有足够上下文或预测坐标越界。

## 模型校验

交付或移动文件后，可在 `API/` 目录执行：

```bash
sha256sum -c model/SHA256SUMS
```

两份权重必须与代码一起保留在 `model/` 目录中。
