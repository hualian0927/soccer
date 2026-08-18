# SoccerNetGSR Local Pipeline Flowchart

下面的 Mermaid 流程图描述当前本地项目从普通足球视频输入，到最终带检测框和右上角战术图的视频输出的完整流程。

## PNG 图片版

![SoccerNetGSR local pipeline flowchart](soccernetgsr_pipeline_flowchart.png)

## Mermaid 可编辑版

```mermaid
flowchart TD
    A["输入视频<br/>soccer_input_dataset/test.mp4"] --> B["本地端到端脚本<br/>run_local_video_gsr_visualization.py"]

    B --> C["抽帧<br/>img1/000001.jpg ..."]
    B --> D["生成本地配置<br/>config.local.yaml"]

    C --> E["球场关键点检测<br/>kpts.py"]
    E --> F["单应性 / 球场映射<br/>每帧 .npy"]

    C --> G["目标检测 + 跟踪 + 角色识别<br/>inference_soccernetGSR.py"]
    D --> G
    G --> H["初始跟踪结果<br/>interpolate_SNGS-999.txt"]

    H --> I["去重检测框<br/>IDATR/rmv_doub_bbox.py"]
    I --> J["生成 tracklets<br/>IDATR/gen_tracklets.py"]
    J --> K["轨迹 refine / 合并拆分<br/>IDATR/refine_tracklets.py"]
    K --> L["精修跟踪结果<br/>refined_SNGS-999.txt"]

    L --> M["映射到球场坐标<br/>IDATR/create_court_file.py"]
    F --> M
    M --> N["球场坐标文件<br/>court_meter_SNGS-999.txt"]

    L --> O["写 SoccerNetGS JSON<br/>write_json_file_team.py"]
    N --> O
    D --> O

    O --> P["足球后处理<br/>面积 / 宽高比 / 持续帧数 / Ball 投票 / 光流一致性"]
    P --> Q["稳定基础 JSON<br/>SNGS-999.json"]

    Q --> R["裁判后处理<br/>refine_referee_roles.py"]
    L --> R
    R --> S["裁判修正版 JSON<br/>SNGS-999.referee_refined.json"]
    R --> T["裁判识别报告<br/>SNGS-999.referee_refined_report.md"]

    S --> U["可视化绘制<br/>visualize_prediction_results.py"]
    C --> U
    D --> U
    U --> V["逐帧可视化结果<br/>visualization_referee_magenta/.../*.jpg"]

    V --> W["合成视频<br/>OpenCV VideoWriter"]
    W --> X["最终输出视频<br/>test_gsr_visualized_full_referee_magenta.mp4"]

    subgraph Legend["颜色 / 标签约定"]
        Y1["左队：红色框和红色战术点"]
        Y2["右队：蓝色框和蓝色战术点"]
        Y3["足球：黄色框和黄色战术点"]
        Y4["裁判：洋红色框和洋红色战术点"]
    end
```

## 简化流程

```mermaid
flowchart LR
    A["视频输入"] --> B["抽帧"]
    B --> C["关键点与球场映射"]
    B --> D["目标检测 / 跟踪 / 角色识别"]
    D --> E["轨迹 refine"]
    C --> F["球场坐标转换"]
    E --> F
    F --> G["JSON 生成"]
    G --> H["足球过滤"]
    H --> I["裁判后处理"]
    I --> J["可视化帧"]
    J --> K["合成最终视频"]
```

## 主要输入输出文件

| 阶段 | 关键文件 | 作用 |
| --- | --- | --- |
| 输入 | `soccer_input_dataset/test.mp4` | 原始足球视频 |
| 本地适配 | `run_local_video_gsr_visualization.py` | 组织数据目录并串联全流程 |
| 抽帧结果 | `soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-999/img1/` | 后续模型逐帧处理的输入 |
| 跟踪结果 | `interpolate_SNGS-999.txt`、`refined_SNGS-999.txt` | 图像坐标中的目标轨迹 |
| 球场坐标 | `court_meter_SNGS-999.txt` | 目标在球场平面上的坐标 |
| 稳定基础结果 | `SNGS-999.json` | 足球光流过滤后的基础 JSON |
| 裁判修正结果 | `SNGS-999.referee_refined.json` | 裁判从两队中分离后的最终 JSON |
| 最终视频 | `test_gsr_visualized_full_referee_magenta.mp4` | 当前推荐可视化输出 |
