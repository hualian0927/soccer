# 目录与逐文件用途索引

更新日期：2026-10-09。主入口和常用文件见 [README](../README.md)。本表覆盖当前 Git 交付文件，不包括权重、视频、缓存及个人调研资料。

工作流统一从仓库根目录通过 `python -m workflows.分类.脚本名` 运行；下方源码链接用于阅读，不表示应直接按路径执行 Python 文件。

上游组件保留既有目录及许可；旧路径迁移后的参数与数据格式不变。

## 根目录

项目安装、说明及版本控制入口。

| 文件 | 作用 |
| --- | --- |
| [.env.example](<../.env.example>) | 后端环境变量模板，不含真实密钥 |
| [.gitignore](<../.gitignore>) | Git 排除规则，保留代码，排除密钥、模型和媒体 |
| [README.md](<../README.md>) | 项目总览、快速开始、目录文件用途与迁移说明 |
| [requirements.txt](<../requirements.txt>) | 完整 Python 管线依赖入口 |
| [setup.py](<../setup.py>) | 上游 YOLOX 安装、版本及扩展编译入口 |

## API/

足球检测与补全接口。

| 文件 | 作用 |
| --- | --- |
| [README.md](<../API/README.md>) | 足球 RF-DETR/SAHI/TCN 子包说明 |
| [complete.py](<../API/complete.py>) | 轨迹补全 |
| [detect.py](<../API/detect.py>) | 检测 |
| [requirement.txt](<../API/requirement.txt>) | 该子包依赖 |
| [run.py](<../API/run.py>) | 足球检测完整入口 |

## API/football_inference/

足球检测与补全接口。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../API/football_inference/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [api.py](<../API/football_inference/api.py>) | 足球检测与补全的 Python 调用接口 |
| [pipeline.py](<../API/football_inference/pipeline.py>) | 足球 RF-DETR 检测、SAHI 切片和 TCN 轨迹补全 |
| [tcn_model.py](<../API/football_inference/tcn_model.py>) | 轨迹补全时序卷积模型定义 |

## IDATR/

轨迹后处理与球场坐标转换。

| 文件 | 作用 |
| --- | --- |
| [Tracklet.py](<../IDATR/Tracklet.py>) | 轨迹对象 |
| [create_court_file.py](<../IDATR/create_court_file.py>) | 球场坐标文件 |
| [gen_tracklets.py](<../IDATR/gen_tracklets.py>) | 轨迹段生成 |
| [refine_tracklets.py](<../IDATR/refine_tracklets.py>) | 轨迹修正 |
| [rmv_doub_bbox.py](<../IDATR/rmv_doub_bbox.py>) | 去重 |

## assets/fonts/

随代码发布的静态资源。

| 文件 | 作用 |
| --- | --- |
| [NotoSansCJKsc-Regular.otf](<../assets/fonts/NotoSansCJKsc-Regular.otf>) | 中文视频、图表与报告字体 |

## configs/

模型和分析配置。

| 文件 | 作用 |
| --- | --- |
| [config.yaml](<../configs/config.yaml>) | 检测、跟踪、资源位置及模型配置 |
| [tactical_analysis.yaml](<../configs/tactical_analysis.yaml>) | 分析器启用配置和算法阈值 |

## docs/

运行文档、技术研究与历史进度记录。

| 文件 | 作用 |
| --- | --- |
| [API_INTEGRATION.md](<../docs/API_INTEGRATION.md>) | 外接模型 |
| [ENVIRONMENT.md](<../docs/ENVIRONMENT.md>) | 环境 |
| [FEATURES.md](<../docs/FEATURES.md>) | 功能边界 |
| [FILE_INDEX.md](<../docs/FILE_INDEX.md>) | 当前已提交目录与逐文件用途索引 |
| [L1事件层初步复现范围.md](<../docs/L1事件层初步复现范围.md>) | 文档：L1 事件层初步复现范围 |
| [L3-L4战术深度复盘与评语.md](<../docs/L3-L4战术深度复盘与评语.md>) | 文档：L3-L4 战术深度复盘与评语 |
| [OpenAI视觉审核工作流.md](<../docs/OpenAI视觉审核工作流.md>) | 文档：OpenAI 时序视觉审核工作流 |
| [RUNNING.md](<../docs/RUNNING.md>) | 运行 |
| [UPSTREAM_README.md](<../docs/UPSTREAM_README.md>) | 文档：原仓库 README 归档 |
| [local_soccernetgsr_changes.md](<../docs/local_soccernetgsr_changes.md>) | 文档：Local SoccerNetGSR Changes |
| [render_tactical_project_flowchart.py](<../docs/render_tactical_project_flowchart.py>) | 流程图生成 |
| [soccernetgsr_pipeline_flowchart.md](<../docs/soccernetgsr_pipeline_flowchart.md>) | 文档：SoccerNetGSR Local Pipeline Flowchart |
| [tactical_analysis_framework.md](<../docs/tactical_analysis_framework.md>) | 文档：基础足球技战术分析框架 |
| [updated_pdf_requirements_mapping.md](<../docs/updated_pdf_requirements_mapping.md>) | 文档：新版足球技战术分析要求实现对照 |
| [中转站视觉审核测试.md](<../docs/中转站视觉审核测试.md>) | 文档：中转站视觉审核测试 |
| [二维投影与视觉模型技战术分析边界.md](<../docs/二维投影与视觉模型技战术分析边界.md>) | 文档：二维投影与视觉模型技战术分析边界 |
| [五分钟助手切片复核版.md](<../docs/五分钟助手切片复核版.md>) | 文档：五分钟助手切片复核版 |
| [全景与对侧视频融合技战术分析调研.md](<../docs/全景与对侧视频融合技战术分析调研.md>) | 文档：全景、对侧视频与人球轨迹融合：技战术分析及拍摄建议 |
| [分层视觉复核与门将改进.md](<../docs/分层视觉复核与门将改进.md>) | 文档：分层视觉复核与门将改进 |
| [双机位球轨迹分析与RFDETR接入.md](<../docs/双机位球轨迹分析与RFDETR接入.md>) | 文档：双机位球轨迹分析与 RF-DETR/TCN 接入 |
| [周报_2026-09-21至09-27_DONE_TODO.md](<../docs/周报_2026-09-21至09-27_DONE_TODO.md>) | 文档：足球技战术分析项目周报：DONE 与 TODO |
| [国内外足球技战术分析平台调研.md](<../docs/国内外足球技战术分析平台调研.md>) | 文档：国内外足球技战术分析网站与 App 调研 |
| [本地环境配置与运行.md](<../docs/本地环境配置与运行.md>) | 文档：SoccerNetGSR 足球技战术分析项目：本地环境配置与运行 |
| [球员编号与个人片段分析.md](<../docs/球员编号与个人片段分析.md>) | 文档：球员编号与个人片段分析 |
| [论文技术借鉴与项目落地总结.md](<../docs/论文技术借鉴与项目落地总结.md>) | 文档：足球技战术分析：论文技术借鉴与项目落地总结 |
| [足球技战术分析进度_DONE_TODO.md](<../docs/足球技战术分析进度_DONE_TODO.md>) | 文档：足球技战术分析项目进度：DONE 与 TODO |
| [阵型可视化与分层分析优化_2026-10-06.md](<../docs/阵型可视化与分层分析优化_2026-10-06.md>) | 文档：阵型可视化与分层分析优化 |

## examples/

可运行的轻量示例。

| 文件 | 作用 |
| --- | --- |
| [structured_demo.py](<../examples/structured_demo.py>) | 无权重、无视频、无 API 的合成 GSR Demo |

## exp/

YOLOX 实验参数。

| 文件 | 作用 |
| --- | --- |
| [yolox_x_soccernet.py](<../exp/yolox_x_soccernet.py>) | SoccerNet YOLOX 实验配置 |

## jersey_model/

球衣身份网络。

| 文件 | 作用 |
| --- | --- |
| [CLIPFinetune.py](<../jersey_model/CLIPFinetune.py>) | 球衣/角色 CLIP 网络 |

## requirements/

按使用场景拆分的环境依赖。

| 文件 | 作用 |
| --- | --- |
| [analysis.txt](<../requirements/analysis.txt>) | 结构化轨迹分析最小依赖 |
| [llama.txt](<../requirements/llama.txt>) | 可选 LLaMA 球衣识别依赖 |

## review_annotations/

开发样例审核标注。

| 文件 | 作用 |
| --- | --- |
| [test_5min_assistant_v3.json](<../review_annotations/test_5min_assistant_v3.json>) | 五分钟样例的事件人工复核记录 |
| [test_5min_formation_frames.json](<../review_annotations/test_5min_formation_frames.json>) | 五分钟样例阵型证据帧记录 |
| [test_5min_layered_v4.json](<../review_annotations/test_5min_layered_v4.json>) | 五分钟样例分层战术评语 |

## scripts/

环境、发布与轨迹批处理工具。

| 文件 | 作用 |
| --- | --- |
| [audit_release.py](<../scripts/audit_release.py>) | 暂存区密钥/大文件检查 |
| [check_environment.py](<../scripts/check_environment.py>) | 只读环境诊断 |
| [refine_tracklets.sh](<../scripts/refine_tracklets.sh>) | 轨迹后处理批处理 |

## sfr/

上游球场分割和标定实现。

| 文件 | 作用 |
| --- | --- |
| [inference.py](<../sfr/inference.py>) | 球场分割推理 |
| [utils_opt.py](<../sfr/utils_opt.py>) | 标定优化 |

## sfr/models/

上游球场分割和标定实现。

| 文件 | 作用 |
| --- | --- |
| [attention_effv2sunet.py](<../sfr/models/attention_effv2sunet.py>) | 网络结构：attention_effv2sunet |
| [non_local_embedded_gaussian.py](<../sfr/models/non_local_embedded_gaussian.py>) | 网络结构：non_local_embedded_gaussian |

## sfr/template/

上游球场分割和标定实现。

| 文件 | 作用 |
| --- | --- |
| [soccernet_template_97.npy](<../sfr/template/soccernet_template_97.npy>) | 球场分割与标定使用的几何模板 |

## sn-trackeval/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [.gitignore](<../sn-trackeval/.gitignore>) | 辅助资源或数据格式说明：.gitignore |
| [LICENSE](<../sn-trackeval/LICENSE>) | 上游版权与许可条款 |
| [Readme.md](<../sn-trackeval/Readme.md>) | 文档：SoccerNet TrackEval |
| [minimum_requirements.txt](<../sn-trackeval/minimum_requirements.txt>) | 依赖、配置或格式定义：minimum_requirements.txt |
| [pyproject.toml](<../sn-trackeval/pyproject.toml>) | 依赖、配置或格式定义：pyproject.toml |
| [setup.py](<../sn-trackeval/setup.py>) | 上游跟踪指标、数据格式与评测实现：setup |

## sn-trackeval/docs/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [BDD100k-format.txt](<../sn-trackeval/docs/BDD100k-format.txt>) | 依赖、配置或格式定义：BDD100k-format.txt |
| [DAVIS-format.txt](<../sn-trackeval/docs/DAVIS-format.txt>) | 依赖、配置或格式定义：DAVIS-format.txt |
| [KITTI-format.txt](<../sn-trackeval/docs/KITTI-format.txt>) | 依赖、配置或格式定义：KITTI-format.txt |
| [MOTChallenge-format.txt](<../sn-trackeval/docs/MOTChallenge-format.txt>) | 依赖、配置或格式定义：MOTChallenge-format.txt |
| [MOTS-format.txt](<../sn-trackeval/docs/MOTS-format.txt>) | 依赖、配置或格式定义：MOTS-format.txt |
| [TAO-format.txt](<../sn-trackeval/docs/TAO-format.txt>) | 依赖、配置或格式定义：TAO-format.txt |
| [YouTube-VIS-format.txt](<../sn-trackeval/docs/YouTube-VIS-format.txt>) | 依赖、配置或格式定义：YouTube-VIS-format.txt |

## sn-trackeval/docs/How_To/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [Add_a_new_metric.md](<../sn-trackeval/docs/How_To/Add_a_new_metric.md>) | 文档：How to add a new or custom family of evaluation metrics to TrackEval |

## sn-trackeval/docs/MOTChallenge-Official/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [Readme.md](<../sn-trackeval/docs/MOTChallenge-Official/Readme.md>) | 文档：MOTChallenge Official Evaluation Kit - Multi-Object Tracking - MOT15, MOT16, MOT17, MOT20 |

## sn-trackeval/docs/OpenWorldTracking-Official/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [Readme.md](<../sn-trackeval/docs/OpenWorldTracking-Official/Readme.md>) | 文档：Opening Up Open-World Tracking - Official Evaluation Code |

## sn-trackeval/docs/RobMOTS-Official/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [Readme.md](<../sn-trackeval/docs/RobMOTS-Official/Readme.md>) | 文档：RobMOTS Official Evaluation Code |

## sn-trackeval/scripts/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [comparison_plots.py](<../sn-trackeval/scripts/comparison_plots.py>) | 评测命令入口：comparison_plots |
| [generate_predictions_soccernet_gs.py](<../sn-trackeval/scripts/generate_predictions_soccernet_gs.py>) | 评测命令入口：generate_predictions_soccernet_gs |
| [run_bdd.py](<../sn-trackeval/scripts/run_bdd.py>) | run_bdd.py |
| [run_burst.py](<../sn-trackeval/scripts/run_burst.py>) | run_burst.py |
| [run_burst_ow.py](<../sn-trackeval/scripts/run_burst_ow.py>) | run_burst_ow.py |
| [run_davis.py](<../sn-trackeval/scripts/run_davis.py>) | run_davis.py |
| [run_headtracking_challenge.py](<../sn-trackeval/scripts/run_headtracking_challenge.py>) | run_mot_challenge.py |
| [run_kitti.py](<../sn-trackeval/scripts/run_kitti.py>) | run_kitti.py |
| [run_kitti_mots.py](<../sn-trackeval/scripts/run_kitti_mots.py>) | run_kitti_mots.py |
| [run_mot_challenge.py](<../sn-trackeval/scripts/run_mot_challenge.py>) | run_mot_challenge.py |
| [run_mots_challenge.py](<../sn-trackeval/scripts/run_mots_challenge.py>) | run_mots.py |
| [run_person_path_22.py](<../sn-trackeval/scripts/run_person_path_22.py>) | run_person_path_22.py |
| [run_rob_mots.py](<../sn-trackeval/scripts/run_rob_mots.py>) | 评测命令入口：run_rob_mots |
| [run_soccernet_gs.py](<../sn-trackeval/scripts/run_soccernet_gs.py>) | run_soccernet_gs.py |
| [run_soccernet_mot.py](<../sn-trackeval/scripts/run_soccernet_mot.py>) | run_soccernet_mot.py |
| [run_tao.py](<../sn-trackeval/scripts/run_tao.py>) | run_tao.py |
| [run_tao_ow.py](<../sn-trackeval/scripts/run_tao_ow.py>) | run_tao.py |
| [run_youtube_vis.py](<../sn-trackeval/scripts/run_youtube_vis.py>) | run_youtube_vis.py |

## sn-trackeval/tests/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [test_all_quick.py](<../sn-trackeval/tests/test_all_quick.py>) | 回归测试：all_quick |
| [test_davis.py](<../sn-trackeval/tests/test_davis.py>) | 回归测试：davis |
| [test_metrics.py](<../sn-trackeval/tests/test_metrics.py>) | 回归测试：metrics |
| [test_mot17.py](<../sn-trackeval/tests/test_mot17.py>) | 回归测试：mot17 |
| [test_mots.py](<../sn-trackeval/tests/test_mots.py>) | 回归测试：mots |

## sn-trackeval/trackeval/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../sn-trackeval/trackeval/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [_timing.py](<../sn-trackeval/trackeval/_timing.py>) | 上游跟踪指标、数据格式与评测实现：_timing |
| [eval.py](<../sn-trackeval/trackeval/eval.py>) | 上游跟踪指标、数据格式与评测实现：eval |
| [plotting.py](<../sn-trackeval/trackeval/plotting.py>) | 上游跟踪指标、数据格式与评测实现：plotting |
| [utils.py](<../sn-trackeval/trackeval/utils.py>) | 上游跟踪指标、数据格式与评测实现：utils |

## sn-trackeval/trackeval/baselines/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../sn-trackeval/trackeval/baselines/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [baseline_utils.py](<../sn-trackeval/trackeval/baselines/baseline_utils.py>) | 评测基线处理：baseline_utils |
| [non_overlap.py](<../sn-trackeval/trackeval/baselines/non_overlap.py>) | Non-Overlap: Code to take in a set of raw detections and produce a set of non-overlapping detections from it. |
| [pascal_colormap.py](<../sn-trackeval/trackeval/baselines/pascal_colormap.py>) | 评测基线处理：pascal_colormap |
| [stp.py](<../sn-trackeval/trackeval/baselines/stp.py>) | STP: Simplest Tracker Possible |
| [thresholder.py](<../sn-trackeval/trackeval/baselines/thresholder.py>) | Thresholder |
| [vizualize.py](<../sn-trackeval/trackeval/baselines/vizualize.py>) | Vizualize: Code which converts .txt rle tracking results into a visual .png format. |

## sn-trackeval/trackeval/datasets/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../sn-trackeval/trackeval/datasets/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [_base_dataset.py](<../sn-trackeval/trackeval/datasets/_base_dataset.py>) | 数据集适配/格式：_base_dataset |
| [bdd100k.py](<../sn-trackeval/trackeval/datasets/bdd100k.py>) | 数据集适配/格式：bdd100k |
| [burst.py](<../sn-trackeval/trackeval/datasets/burst.py>) | 数据集适配/格式：burst |
| [burst_ow.py](<../sn-trackeval/trackeval/datasets/burst_ow.py>) | 数据集适配/格式：burst_ow |
| [davis.py](<../sn-trackeval/trackeval/datasets/davis.py>) | 数据集适配/格式：davis |
| [head_tracking_challenge.py](<../sn-trackeval/trackeval/datasets/head_tracking_challenge.py>) | 数据集适配/格式：head_tracking_challenge |
| [kitti_2d_box.py](<../sn-trackeval/trackeval/datasets/kitti_2d_box.py>) | 数据集适配/格式：kitti_2d_box |
| [kitti_mots.py](<../sn-trackeval/trackeval/datasets/kitti_mots.py>) | 数据集适配/格式：kitti_mots |
| [mot_challenge_2d_box.py](<../sn-trackeval/trackeval/datasets/mot_challenge_2d_box.py>) | 数据集适配/格式：mot_challenge_2d_box |
| [mots_challenge.py](<../sn-trackeval/trackeval/datasets/mots_challenge.py>) | 数据集适配/格式：mots_challenge |
| [person_path_22.py](<../sn-trackeval/trackeval/datasets/person_path_22.py>) | 数据集适配/格式：person_path_22 |
| [rob_mots.py](<../sn-trackeval/trackeval/datasets/rob_mots.py>) | 数据集适配/格式：rob_mots |
| [rob_mots_classmap.py](<../sn-trackeval/trackeval/datasets/rob_mots_classmap.py>) | 数据集适配/格式：rob_mots_classmap |
| [run_rob_mots.py](<../sn-trackeval/trackeval/datasets/run_rob_mots.py>) | 数据集适配/格式：run_rob_mots |
| [soccernet_gs.py](<../sn-trackeval/trackeval/datasets/soccernet_gs.py>) | 数据集适配/格式：soccernet_gs |
| [tao.py](<../sn-trackeval/trackeval/datasets/tao.py>) | 数据集适配/格式：tao |
| [tao_ow.py](<../sn-trackeval/trackeval/datasets/tao_ow.py>) | 数据集适配/格式：tao_ow |
| [youtube_vis.py](<../sn-trackeval/trackeval/datasets/youtube_vis.py>) | 数据集适配/格式：youtube_vis |

## sn-trackeval/trackeval/datasets/burst_helpers/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [BURST_SPECIFIC_ISSUES.md](<../sn-trackeval/trackeval/datasets/burst_helpers/BURST_SPECIFIC_ISSUES.md>) | 文档：BURST_SPECIFIC_ISSUES |
| [__init__.py](<../sn-trackeval/trackeval/datasets/burst_helpers/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [burst_base.py](<../sn-trackeval/trackeval/datasets/burst_helpers/burst_base.py>) | 数据集适配/格式：burst_base |
| [burst_ow_base.py](<../sn-trackeval/trackeval/datasets/burst_helpers/burst_ow_base.py>) | 数据集适配/格式：burst_ow_base |
| [convert_burst_format_to_tao_format.py](<../sn-trackeval/trackeval/datasets/burst_helpers/convert_burst_format_to_tao_format.py>) | 数据集适配/格式：convert_burst_format_to_tao_format |
| [format_converter.py](<../sn-trackeval/trackeval/datasets/burst_helpers/format_converter.py>) | 数据集适配/格式：format_converter |
| [tao_categories.json](<../sn-trackeval/trackeval/datasets/burst_helpers/tao_categories.json>) | 数据集适配/格式：tao_categories |

## sn-trackeval/trackeval/metrics/

上游跟踪指标、数据格式与评测实现。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../sn-trackeval/trackeval/metrics/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [_base_metric.py](<../sn-trackeval/trackeval/metrics/_base_metric.py>) | 跟踪评测指标：_base_metric |
| [clear.py](<../sn-trackeval/trackeval/metrics/clear.py>) | 跟踪评测指标：clear |
| [count.py](<../sn-trackeval/trackeval/metrics/count.py>) | 跟踪评测指标：count |
| [hota.py](<../sn-trackeval/trackeval/metrics/hota.py>) | 跟踪评测指标：hota |
| [identity.py](<../sn-trackeval/trackeval/metrics/identity.py>) | 跟踪评测指标：identity |
| [ideucl.py](<../sn-trackeval/trackeval/metrics/ideucl.py>) | 跟踪评测指标：ideucl |
| [j_and_f.py](<../sn-trackeval/trackeval/metrics/j_and_f.py>) | 跟踪评测指标：j_and_f |
| [track_map.py](<../sn-trackeval/trackeval/metrics/track_map.py>) | 跟踪评测指标：track_map |
| [vace.py](<../sn-trackeval/trackeval/metrics/vace.py>) | 跟踪评测指标：vace |

## tactical_analysis/

可复用的技战术分析库。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../tactical_analysis/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [base.py](<../tactical_analysis/base.py>) | 数据契约和分析器接口 |
| [features.py](<../tactical_analysis/features.py>) | 时序特征 |
| [formation_evidence.py](<../tactical_analysis/formation_evidence.py>) | 阵型证据几何 |
| [gsr_io.py](<../tactical_analysis/gsr_io.py>) | 读取 GSR |
| [l1.py](<../tactical_analysis/l1.py>) | Unified L1 event catalog derived from evidence-first analyzer outputs. |
| [models.py](<../tactical_analysis/models.py>) | Shared data contracts used by every tactical-analysis stage. |
| [openai_review.py](<../tactical_analysis/openai_review.py>) | API 请求、预算与缓存 |
| [pipeline.py](<../tactical_analysis/pipeline.py>) | 统一分析器调度、容错和报告输出 |
| [player_focus.py](<../tactical_analysis/player_focus.py>) | 编号跟踪和个人片段 |
| [player_tracker.py](<../tactical_analysis/player_tracker.py>) | Conservative jersey-color constraints around the existing BoT-SORT engine. |
| [projection_l1.py](<../tactical_analysis/projection_l1.py>) | 事件转换 |
| [reporting.py](<../tactical_analysis/reporting.py>) | 报告、高光和摘要 |
| [reviewed_layers.py](<../tactical_analysis/reviewed_layers.py>) | 复核结果分层 |

## tactical_analysis/analyzers/

可复用的技战术分析库。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../tactical_analysis/analyzers/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [attacking_chains.py](<../tactical_analysis/analyzers/attacking_chains.py>) | 进攻链 |
| [ball_out.py](<../tactical_analysis/analyzers/ball_out.py>) | 出界 |
| [defensive_gaps.py](<../tactical_analysis/analyzers/defensive_gaps.py>) | 空档 |
| [defensive_interventions.py](<../tactical_analysis/analyzers/defensive_interventions.py>) | 防守干预 |
| [defensive_risk.py](<../tactical_analysis/analyzers/defensive_risk.py>) | 防守风险 |
| [events.py](<../tactical_analysis/analyzers/events.py>) | 基础时序事件 |
| [formation.py](<../tactical_analysis/analyzers/formation.py>) | 阵型 |
| [goalkeeper.py](<../tactical_analysis/analyzers/goalkeeper.py>) | 门将候选 |
| [numerical_superiority.py](<../tactical_analysis/analyzers/numerical_superiority.py>) | 人数优势 |
| [pass_network.py](<../tactical_analysis/analyzers/pass_network.py>) | 传球网络 |
| [phases.py](<../tactical_analysis/analyzers/phases.py>) | 比赛阶段 |
| [pressing.py](<../tactical_analysis/analyzers/pressing.py>) | 压迫 |
| [progression.py](<../tactical_analysis/analyzers/progression.py>) | 推进 |
| [quality.py](<../tactical_analysis/analyzers/quality.py>) | 质量门控 |
| [relations.py](<../tactical_analysis/analyzers/relations.py>) | 局部关系 |
| [set_pieces.py](<../tactical_analysis/analyzers/set_pieces.py>) | 定位球 |
| [spatial.py](<../tactical_analysis/analyzers/spatial.py>) | 空间统计 |
| [team_shape.py](<../tactical_analysis/analyzers/team_shape.py>) | 宽度/纵深/线距 |
| [transitions.py](<../tactical_analysis/analyzers/transitions.py>) | 转换 |

## template/

球场几何模板。

| 文件 | 作用 |
| --- | --- |
| [soccernet_template_97.npy](<../template/soccernet_template_97.npy>) | SoccerNet 球场关键点模板 |

## tests/

项目功能和目录迁移回归测试。

| 文件 | 作用 |
| --- | --- |
| [test_analyze_full_pitch_projection.py](<../tests/test_analyze_full_pitch_projection.py>) | 回归测试：analyze_full_pitch_projection |
| [test_analyze_projection_folder.py](<../tests/test_analyze_projection_folder.py>) | 回归测试：analyze_projection_folder |
| [test_assistant_review_version.py](<../tests/test_assistant_review_version.py>) | 回归测试：assistant_review_version |
| [test_formation_evidence.py](<../tests/test_formation_evidence.py>) | 回归测试：formation_evidence |
| [test_layered_tactical_review.py](<../tests/test_layered_tactical_review.py>) | 回归测试：layered_tactical_review |
| [test_make_vision_review_video.py](<../tests/test_make_vision_review_video.py>) | 回归测试：make_vision_review_video |
| [test_openai_tactical_review.py](<../tests/test_openai_tactical_review.py>) | 回归测试：openai_tactical_review |
| [test_player_focus.py](<../tests/test_player_focus.py>) | 回归测试：player_focus |
| [test_projection_ball_and_adapter.py](<../tests/test_projection_ball_and_adapter.py>) | 回归测试：projection_ball_and_adapter |
| [test_projection_l1.py](<../tests/test_projection_l1.py>) | 回归测试：projection_l1 |
| [test_reviewed_layers.py](<../tests/test_reviewed_layers.py>) | 回归测试：reviewed_layers |
| [test_tactical_analysis_extensions.py](<../tests/test_tactical_analysis_extensions.py>) | 回归测试：tactical_analysis_extensions |
| [test_tactical_analysis_framework.py](<../tests/test_tactical_analysis_framework.py>) | 回归测试：tactical_analysis_framework |
| [test_team_identity.py](<../tests/test_team_identity.py>) | 回归测试：team_identity |
| [test_vision_candidate_review.py](<../tests/test_vision_candidate_review.py>) | 回归测试：vision_candidate_review |
| [test_workflow_layout.py](<../tests/test_workflow_layout.py>) | 回归测试：workflow_layout |

## utils/

上游通用辅助函数。

| 文件 | 作用 |
| --- | --- |
| [bbox.py](<../utils/bbox.py>) | 框几何 |
| [sys_utils.py](<../utils/sys_utils.py>) | 张量/系统辅助 |
| [transforms.py](<../utils/transforms.py>) | 图像变换 |

## web_preview/

React 本地 Demo 与 Node 本地后端。

| 文件 | 作用 |
| --- | --- |
| [.gitignore](<../web_preview/.gitignore>) | 辅助资源或数据格式说明：.gitignore |
| [README.md](<../web_preview/README.md>) | 网页运行说明 |
| [index.html](<../web_preview/index.html>) | 网页入口 |
| [package-lock.json](<../web_preview/package-lock.json>) | 前端依赖 |
| [package.json](<../web_preview/package.json>) | 依赖、配置或格式定义：package.json |
| [scripts_extract_posters.py](<../web_preview/scripts_extract_posters.py>) | 封面提取 |
| [vite.config.js](<../web_preview/vite.config.js>) | 开发服务与本地接口挂载 |

## web_preview/scripts/

React 本地 Demo 与 Node 本地后端。

| 文件 | 作用 |
| --- | --- |
| [test-review-state.mjs](<../web_preview/scripts/test-review-state.mjs>) | 状态单测 |
| [verify-formation-focus.mjs](<../web_preview/scripts/verify-formation-focus.mjs>) | 阵型交互回归 |
| [verify-openai.mjs](<../web_preview/scripts/verify-openai.mjs>) | 审核结果页面回归 |
| [verify-player-focus.mjs](<../web_preview/scripts/verify-player-focus.mjs>) | 个人选段回归 |
| [verify-preview.mjs](<../web_preview/scripts/verify-preview.mjs>) | 页面回归 |

## web_preview/server/

React 本地 Demo 与 Node 本地后端。

| 文件 | 作用 |
| --- | --- |
| [local-analysis.js](<../web_preview/server/local-analysis.js>) | 上传、分析队列、报告和媒体接口 |
| [player-focus.js](<../web_preview/server/player-focus.js>) | 个人选段导出接口 |

## web_preview/src/

React 本地 Demo 与 Node 本地后端。

| 文件 | 作用 |
| --- | --- |
| [App.jsx](<../web_preview/src/App.jsx>) | 素材、项目、分析和播放页面 |
| [FormationEvidence.jsx](<../web_preview/src/FormationEvidence.jsx>) | 阵型证据 |
| [LayeredAnalysisPanel.jsx](<../web_preview/src/LayeredAnalysisPanel.jsx>) | 分层评语 |
| [PlayerFocus.jsx](<../web_preview/src/PlayerFocus.jsx>) | 人物选段 |
| [SpatialAnalysisPage.jsx](<../web_preview/src/SpatialAnalysisPage.jsx>) | 空间页 |
| [SwitchableCameraStage.jsx](<../web_preview/src/SwitchableCameraStage.jsx>) | 双机位切换 |
| [data.js](<../web_preview/src/data.js>) | 样例项目元数据 |
| [formation-evidence.css](<../web_preview/src/formation-evidence.css>) | 网页样式：formation-evidence |
| [main.jsx](<../web_preview/src/main.jsx>) | 挂载入口 |
| [player-focus.css](<../web_preview/src/player-focus.css>) | 网页样式：player-focus |
| [review-state.js](<../web_preview/src/review-state.js>) | 复核状态 |
| [styles.css](<../web_preview/src/styles.css>) | 网页样式：styles |

## workflows/

按功能分类的命令行入口及其辅助模块。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../workflows/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |

## workflows/data/

按功能分类的命令行入口及其辅助模块。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../workflows/data/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [download_properties.py](<../workflows/data/download_properties.py>) | 按配置下载模型资源，下载权重不提交 |

## workflows/gsr/

按功能分类的命令行入口及其辅助模块。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../workflows/gsr/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [ball_detection_adapter.py](<../workflows/gsr/ball_detection_adapter.py>) | 足球检测结果适配与轨迹合并 |
| [inference_soccernetGSR.py](<../workflows/gsr/inference_soccernetGSR.py>) | 检测、跟踪及角色识别 |
| [kpts.py](<../workflows/gsr/kpts.py>) | 球场关键点与投影 |
| [run_fast_long_video_gsr.py](<../workflows/gsr/run_fast_long_video_gsr.py>) | 长视频轻量预览 |
| [run_local_video_gsr_visualization.py](<../workflows/gsr/run_local_video_gsr_visualization.py>) | 视频到 GSR 的完整管线 |
| [write_json_file_team.py](<../workflows/gsr/write_json_file_team.py>) | 导出统一比赛状态 JSON |

## workflows/identity/

按功能分类的命令行入口及其辅助模块。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../workflows/identity/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [jersey_color.py](<../workflows/identity/jersey_color.py>) | 球衣颜色特征 |
| [refine_gsr_video_identities.py](<../workflows/identity/refine_gsr_video_identities.py>) | 结合视频修正 GSR 身份 |
| [refine_referee_roles.py](<../workflows/identity/refine_referee_roles.py>) | 裁判角色修正及重绘 |
| [refine_team_identities.py](<../workflows/identity/refine_team_identities.py>) | 跟踪文件的球队修正 |
| [team_identity.py](<../workflows/identity/team_identity.py>) | 轨迹身份推断 |
| [tracklet_appearance.py](<../workflows/identity/tracklet_appearance.py>) | CLIP 外观复核 |

## workflows/players/

按功能分类的命令行入口及其辅助模块。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../workflows/players/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [build_player_focus_video.py](<../workflows/players/build_player_focus_video.py>) | 生成可开关的球员编号和轨迹 |
| [export_player_focus.py](<../workflows/players/export_player_focus.py>) | 指定球员时间段的短片与报告 |

## workflows/projection/

按功能分类的命令行入口及其辅助模块。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../workflows/projection/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [analyze_full_pitch_projection.py](<../workflows/projection/analyze_full_pitch_projection.py>) | 全场位置与空间统计 |
| [analyze_projection_folder.py](<../workflows/projection/analyze_projection_folder.py>) | 成组处理素材并生成报告 |
| [analyze_projection_with_ball.py](<../workflows/projection/analyze_projection_with_ball.py>) | 加入足球、边界及重启候选 |

## workflows/review/

按功能分类的命令行入口及其辅助模块。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../workflows/review/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [build_assistant_review_version.py](<../workflows/review/build_assistant_review_version.py>) | 合并已有审核标注 |
| [build_layered_tactical_review.py](<../workflows/review/build_layered_tactical_review.py>) | 生成分层分析包及可选 API 评语 |
| [build_reviewed_tactical_layers.py](<../workflows/review/build_reviewed_tactical_layers.py>) | 将已审核评语补入层级 |
| [prepare_assistant_review_sheets.py](<../workflows/review/prepare_assistant_review_sheets.py>) | 抽帧审核拼图 |
| [prepare_layered_review_evidence.py](<../workflows/review/prepare_layered_review_evidence.py>) | 分层节点证据 |
| [review_tactical_candidates_with_vision.py](<../workflows/review/review_tactical_candidates_with_vision.py>) | 候选视觉复核 |
| [run_openai_tactical_review.py](<../workflows/review/run_openai_tactical_review.py>) | Responses API 时序复核 |

## workflows/tactical/

按功能分类的命令行入口及其辅助模块。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../workflows/tactical/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [analyze_tactical_metrics.py](<../workflows/tactical/analyze_tactical_metrics.py>) | 早期基础指标统计 |
| [run_match_analysis_workflow.py](<../workflows/tactical/run_match_analysis_workflow.py>) | 长比赛镜头扫描、远近景分流与分段处理 |
| [run_tactical_analysis.py](<../workflows/tactical/run_tactical_analysis.py>) | 调用统一分析器输出报告 |

## workflows/visualization/

按功能分类的命令行入口及其辅助模块。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../workflows/visualization/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [build_formation_evidence.py](<../workflows/visualization/build_formation_evidence.py>) | 阵型原图、框线图及短片 |
| [export_set_piece_clips.py](<../workflows/visualization/export_set_piece_clips.py>) | 定位球分类切片 |
| [export_tactical_highlights.py](<../workflows/visualization/export_tactical_highlights.py>) | 高光裁剪 |
| [make_formation_analysis_video.py](<../workflows/visualization/make_formation_analysis_video.py>) | 阵型视频 |
| [make_individual_technique_analysis_video.py](<../workflows/visualization/make_individual_technique_analysis_video.py>) | 射门/门将姿态窗口 |
| [make_offensive_analysis_video.py](<../workflows/visualization/make_offensive_analysis_video.py>) | 进攻可视化 |
| [make_tactical_report_video.py](<../workflows/visualization/make_tactical_report_video.py>) | 插入报告结论 |
| [make_tactical_visualization_video.py](<../workflows/visualization/make_tactical_visualization_video.py>) | 检测框、战术板或仅框视频 |
| [make_vision_review_video.py](<../workflows/visualization/make_vision_review_video.py>) | 视觉复核结果展示 |
| [render_projection_l1_video.py](<../workflows/visualization/render_projection_l1_video.py>) | 二维事件视频 |
| [visualize_local_result.py](<../workflows/visualization/visualize_local_result.py>) | 本地结果绘制入口 |
| [visualize_prediction_results.py](<../workflows/visualization/visualize_prediction_results.py>) | 原始预测绘制 |
| [visualize_referee_distinct_colors.py](<../workflows/visualization/visualize_referee_distinct_colors.py>) | 裁判独立配色 |

## yolox/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../yolox/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |

## yolox/EIoU_tracker/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [Deep_EIoU.py](<../yolox/EIoU_tracker/Deep_EIoU.py>) | EIoU 关联跟踪：Deep_EIoU |
| [Deep_EIoU_pitch.py](<../yolox/EIoU_tracker/Deep_EIoU_pitch.py>) | EIoU 关联跟踪：Deep_EIoU_pitch |
| [Deep_EIoU_pitch2.py](<../yolox/EIoU_tracker/Deep_EIoU_pitch2.py>) | EIoU 关联跟踪：Deep_EIoU_pitch2 |
| [KF.py](<../yolox/EIoU_tracker/KF.py>) | EIoU 关联跟踪：KF |
| [__init__.py](<../yolox/EIoU_tracker/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [basetrack.py](<../yolox/EIoU_tracker/basetrack.py>) | EIoU 关联跟踪：basetrack |
| [kalman_filter.py](<../yolox/EIoU_tracker/kalman_filter.py>) | EIoU 关联跟踪：kalman_filter |
| [matching.py](<../yolox/EIoU_tracker/matching.py>) | EIoU 关联跟踪：matching |
| [reid_model.py](<../yolox/EIoU_tracker/reid_model.py>) | EIoU 关联跟踪：reid_model |
| [transforms.py](<../yolox/EIoU_tracker/transforms.py>) | EIoU 关联跟踪：transforms |

## yolox/EIoU_tracker/tracking_utils/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../yolox/EIoU_tracker/tracking_utils/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [evaluation.py](<../yolox/EIoU_tracker/tracking_utils/evaluation.py>) | 跟踪辅助工具：evaluation |
| [io.py](<../yolox/EIoU_tracker/tracking_utils/io.py>) | 跟踪辅助工具：io |
| [timer.py](<../yolox/EIoU_tracker/tracking_utils/timer.py>) | 跟踪辅助工具：timer |

## yolox/byte_tracker/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../yolox/byte_tracker/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [association.py](<../yolox/byte_tracker/association.py>) | ByteTrack 关联跟踪：association |
| [basetrack.py](<../yolox/byte_tracker/basetrack.py>) | ByteTrack 关联跟踪：basetrack |
| [byte_iou_tracker.py](<../yolox/byte_tracker/byte_iou_tracker.py>) | ByteTrack 关联跟踪：byte_iou_tracker |
| [byte_tracker.py](<../yolox/byte_tracker/byte_tracker.py>) | ByteTrack 关联跟踪：byte_tracker |
| [kalman_filter.py](<../yolox/byte_tracker/kalman_filter.py>) | ByteTrack 关联跟踪：kalman_filter |
| [matching.py](<../yolox/byte_tracker/matching.py>) | ByteTrack 关联跟踪：matching |
| [reid_model.py](<../yolox/byte_tracker/reid_model.py>) | ByteTrack 关联跟踪：reid_model |

## yolox/data/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../yolox/data/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [data_augment.py](<../yolox/data/data_augment.py>) | Data augmentation functionality. Passed as callable transformations to |
| [data_prefetcher.py](<../yolox/data/data_prefetcher.py>) | 预处理与数据加载：data_prefetcher |
| [dataloading.py](<../yolox/data/dataloading.py>) | 预处理与数据加载：dataloading |
| [samplers.py](<../yolox/data/samplers.py>) | 预处理与数据加载：samplers |

## yolox/data/datasets/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../yolox/data/datasets/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [datasets_wrapper.py](<../yolox/data/datasets/datasets_wrapper.py>) | 数据集适配/格式：datasets_wrapper |
| [mosaicdetection.py](<../yolox/data/datasets/mosaicdetection.py>) | 数据集适配/格式：mosaicdetection |
| [mot.py](<../yolox/data/datasets/mot.py>) | 数据集适配/格式：mot |

## yolox/evaluators/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../yolox/evaluators/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [coco_evaluator.py](<../yolox/evaluators/coco_evaluator.py>) | 检测与跟踪评测：coco_evaluator |
| [evaluation.py](<../yolox/evaluators/evaluation.py>) | 检测与跟踪评测：evaluation |
| [mot_evaluator.py](<../yolox/evaluators/mot_evaluator.py>) | 检测与跟踪评测：mot_evaluator |

## yolox/exp/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../yolox/exp/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [base_exp.py](<../yolox/exp/base_exp.py>) | 模型实验配置：base_exp |
| [build.py](<../yolox/exp/build.py>) | 模型实验配置：build |
| [yolox_base.py](<../yolox/exp/yolox_base.py>) | 模型实验配置：yolox_base |

## yolox/layers/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../yolox/layers/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [fast_coco_eval_api.py](<../yolox/layers/fast_coco_eval_api.py>) | 检测计算算子：fast_coco_eval_api |

## yolox/layers/csrc/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [vision.cpp](<../yolox/layers/csrc/vision.cpp>) | 检测计算算子：vision |

## yolox/layers/csrc/cocoeval/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [cocoeval.cpp](<../yolox/layers/csrc/cocoeval/cocoeval.cpp>) | 检测计算算子：cocoeval |
| [cocoeval.h](<../yolox/layers/csrc/cocoeval/cocoeval.h>) | 检测计算算子：cocoeval |

## yolox/models/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../yolox/models/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [darknet.py](<../yolox/models/darknet.py>) | 网络结构：darknet |
| [losses.py](<../yolox/models/losses.py>) | 网络结构：losses |
| [network_blocks.py](<../yolox/models/network_blocks.py>) | 网络结构：network_blocks |
| [yolo_fpn.py](<../yolox/models/yolo_fpn.py>) | 网络结构：yolo_fpn |
| [yolo_head.py](<../yolox/models/yolo_head.py>) | 网络结构：yolo_head |
| [yolo_pafpn.py](<../yolox/models/yolo_pafpn.py>) | 网络结构：yolo_pafpn |
| [yolox.py](<../yolox/models/yolox.py>) | 网络结构：yolox |

## yolox/tracking_utils/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../yolox/tracking_utils/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [evaluation.py](<../yolox/tracking_utils/evaluation.py>) | 跟踪辅助工具：evaluation |
| [io.py](<../yolox/tracking_utils/io.py>) | 跟踪辅助工具：io |
| [timer.py](<../yolox/tracking_utils/timer.py>) | 跟踪辅助工具：timer |

## yolox/utils/

上游目标检测与跟踪框架。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](<../yolox/utils/__init__.py>) | 包初始化与公开接口导出（工作流包只作声明） |
| [allreduce_norm.py](<../yolox/utils/allreduce_norm.py>) | 通用辅助工具：allreduce_norm |
| [boxes.py](<../yolox/utils/boxes.py>) | 通用辅助工具：boxes |
| [checkpoint.py](<../yolox/utils/checkpoint.py>) | 通用辅助工具：checkpoint |
| [demo_utils.py](<../yolox/utils/demo_utils.py>) | 通用辅助工具：demo_utils |
| [dist.py](<../yolox/utils/dist.py>) | This file contains primitives for multi-gpu communication. |
| [ema.py](<../yolox/utils/ema.py>) | 通用辅助工具：ema |
| [logger.py](<../yolox/utils/logger.py>) | 通用辅助工具：logger |
| [lr_scheduler.py](<../yolox/utils/lr_scheduler.py>) | 通用辅助工具：lr_scheduler |
| [metric.py](<../yolox/utils/metric.py>) | 通用辅助工具：metric |
| [model_utils.py](<../yolox/utils/model_utils.py>) | 通用辅助工具：model_utils |
| [setup_env.py](<../yolox/utils/setup_env.py>) | 通用辅助工具：setup_env |
| [transforms.py](<../yolox/utils/transforms.py>) | 通用辅助工具：transforms |
| [visualize.py](<../yolox/utils/visualize.py>) | 通用辅助工具：visualize |
