# 原仓库 README 归档

本文件保留此前 README 的上游介绍与历史运行说明，路径以原仓库根目录为基准；本地扩展的当前入口请看 [项目 README](../README.md)。历史奖项、模型效果描述来自原文，不是本地扩展的评测结果。

# ⚽ Broadcast2Pitch: Game State Reconstruction from Unconstrained Soccer Videos ⚽

## 🎉 WACV 2026 Best Paper: Application <br> 🏆 Winner of the SoccerNet Game State Reconstruction 2025 Challenge

| Baseline Method | Our Method |
| :---: | :---: |
| <img src="vis/baseline.gif" width="100%"> | <img src="vis/ours.gif" width="100%"> |

**Affiliation:** Human Data Intelligence Lab, Korea Institute of Science and Technology (KIST), Seoul, Republic of Korea

This repository contains the official implementation for the SoccerNet Game State Reconstruction (GSR) task. It includes modules for camera calibration (homography estimation), player tracking, re-identification (ReID), role/jersey number recognition using CLIP and LLaMA models, and tracklet refinement.

> [!NOTE]
> The homography estimation (Sports Field Registration) approach in this repository follows our WACV paper approach, which differs slightly from the method described in [arXiv paper](https://arxiv.org/pdf/2508.19182).

## Installation

### Prerequisites
- Linux
- NVIDIA GPU + CUDA
- Python 3.10（当前 `sports` 环境验证版本）
- Conda

完整的 WSL2、CUDA、Conda、模型目录和网页 Demo 配置步骤见 [本地环境配置与运行](docs/本地环境配置与运行.md)。

### Step-by-Step Installation

1.  **Create a Virtual Environment**
    ```bash
    conda create -n sports python=3.10
    conda activate sports
    ```

2.  **Install PyTorch**
    Install PyTorch compatible with your CUDA version. For example (CUDA 12.1):
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

3.  **Install Dependencies**
    ```bash
    # Install build dependencies first
    pip install numpy cython

    # Install main requirements
    pip install -r requirements.txt
    ```

    *Note: Some packages might need to be installed from source with specific flags:*
    ```bash
    # Install CLIP from source
    pip install git+https://github.com/openai/CLIP.git

    # Install packages requiring numpy/cython during build
    pip install git+https://github.com/KaiyangZhou/deep-person-reid.git --no-build-isolation
    pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' --no-build-isolation
    ```

4.  **Setup Project**
    ```bash
    python setup.py develop
    ```

## Data Preparation

Ensure your data is organized as follows:
```
data/
  SoccerNetGS/
    test/
    challenge/
```
Update `configs/config.yaml` with the correct `DATA_DIR`.

## Usage

### Local Tactical Analysis Framework

The local extension includes a modular, visual-only tactical-analysis layer on top of SoccerNetGSR output. Its 18 analyzers cover the P0-P2 product chain plus four P3 candidates: sustained numerical superiority, high pressing/PPDA, defensive gaps, and transition speed. It produces structured JSON, a concise Chinese report, CSV event/highlight indexes, and an interleaved review video:

```bash
python run_tactical_analysis.py \
    --json-path soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-1000/SNGS-1000.json \
    --input-video soccer_input_dataset/test_5min.mp4 \
    --output-dir soccer_input_dataset/outputs/tactical_framework_test_5min
```

Render the conclusions at their evidence timestamps:

```bash
python make_tactical_report_video.py \
    --input-video soccer_input_dataset/outputs/test_5min_current/test_5min_tactical_base_clean_v2.mp4 \
    --report-json soccer_input_dataset/outputs/tactical_framework_test_5min/tactical_analysis_report.json \
    --output-video soccer_input_dataset/outputs/tactical_framework_test_5min/tactical_review.mp4 \
    --interleave-chapters
```

See [the tactical-analysis framework guide](docs/tactical_analysis_framework.md) for the priority chain, paper-inspired design, module boundaries, and extension process.

### Local Web Demo

The React/Vite preview provides video upload, project organization, an L1 event timeline, manual event review, set-piece type counts, and one playable evidence clip per set piece:

```bash
cd web_preview
npm ci
npm run dev
```

Open `http://localhost:4173/`. Model weights and input/output videos are intentionally not included in this repository.

### Step 1: Download Weights
Download the required model weights from Google Drive.
```bash
python download_properties.py
```

### Step 2: Homography Estimation
Run `kpts.py` to generate homography matrices (`.npy` files) for the video frames.
```bash
python kpts.py
```

### Step 3: Inference (Tracking & Recognition)
Run the main inference script to perform player tracking, ReID, and jersey number/role recognition.
You can choose between **CLIP** (faster) and **LLAMA** (more accurate) for jersey recognition by setting `JERSEY_MODE` in `configs/config.yaml`.

```bash
python inference_soccernetGSR.py
```

### Step 4: Tracking Results Post Processing using IDATR
Perform ID-Aware Tracklets Refinement (IDATR) using the initial tracklet results from step 3.
```bash
./refine_tracklets.sh
```

### Step 5: Format Conversion
Convert the inference results into the required JSON format for evaluation.
```bash
python write_json_file_team.py
```

### Step 6: Evaluation
Use the `sn-trackeval` toolkit to evaluate the results.
```bash
python ./sn-trackeval/scripts/run_soccernet_gs.py \
    --GT_FOLDER /path/to/SoccerNetGS \
    --TRACKERS_FOLDER /path/to/Pred_SN_GS2025/ \
    --TRACKER_SUB_FOLDER "" \
    --SPLIT_TO_EVAL "test"
```

### Step 7: Visualize Prediction Results
Run the script to visualize the generated JSON files:

```bash
python visualize_prediction_results.py --json_dir results/predicted_SNGamestate_results/test
```

## Models

Ensure you have the following checkpoints in the `checkpoints/` directory:
- YOLOX: `yolox_soccernet.pth.tar`
- ReID (OSNet): `sports_model.pth.tar-60`
- SFR (Keypoints): `SoccernetGSR_EfficientNet_Best.pth`
- CLIP Jersey Model: `CLIP_Jersey.pth`
- LLaMA Model: `lora_model_jersey_role_soccernet`

## Acknowledgements

This project builds upon:
- [SoccerNet](https://www.soccer-net.org/)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [DeepEIOU](https://github.com/hsiangwei0903/Deep-EIoU)
- [GTA](https://github.com/sjc042/gta-link)
- [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- [CLIP](https://github.com/openai/CLIP)

## Contact

For inquiries about our work, please contact Jinwook Kim <jinwook.kim21@gmail.com>.
