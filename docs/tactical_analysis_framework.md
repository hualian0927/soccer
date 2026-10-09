# 基础足球技战术分析框架

## 1. 建设目标

这套框架把项目中已有的 GSR、阵型、进攻、个人射门和门将分析能力组织成一条可扩展的优先级链：

1. P0 数据质量：先判断球、球员、球队标签和场地坐标是否足以支撑分析。
2. P0 事件时间轴：输出触球、接球、带球、传球、球权转换、射门和角球候选，形成可回溯事件链。
3. P0 空间结构：输出区域占用、平均站位、宽度、纵深和队形面积。
4. P1 Team Shape：按持球/无球分别统计宽度、纵深、三线位置、线间距、重心高度和凸包面积。
5. P1 阵型倾向：在多个有效远景帧上聚合 4-3-3、4-2-3-1 等结构，不以单帧冒充整场阵型。
6. P1 定位球与门将：追踪角球/任意球的开出和首次稳定落点，并将射门与门将移动、接近足球和后续控制连接为干预候选。
7. P1 比赛阶段：输出后场组织、中场推进、前场进攻、长传和定位球等阶段，并给出夺回、持球、丢失、无球四状态时长。
8. P2 推进与穿线：输出推进传球、三区进入、肋部进入、转移、传中、直塞、Packing 和穿线候选。
9. P2 进攻链路与防守风险：回溯射门前动作，并按进入、穿线、传中、持球推进和射门聚合危险通道。
10. P3 压迫分析：输出高/中/低位压迫、压迫方向、触发情景、参与人数、PPDA 代理和压迫结果。
11. P3 攻防转换：分析前 5 秒推进、转换速度、参攻/回追人数、反抢夺回和丢球后阵型恢复时间。
12. P2 传球网络：输出有向连接、节点中心性、网络连通性与归一化传球熵。
13. P2 机会质量：在距离和角度之外加入防守压力、封堵球员和门将位置特征。
14. P3 空间机会与风险：输出持续动态多打少和防守空档，并关联前场进入、射门等后续结果窗口。
15. 产品交付：同时输出结构化报告、中文比赛摘要、事件时间轴、高光清单、可裁剪片段和按原时间穿插的讲解视频。

框架坚持“候选发现 + 证据 + 置信度 + 使用边界”。当前启发式指标不会被描述成职业数据供应商级真值。

## 2. 论文技术借鉴

| 论文 | 可借鉴优势 | 本次落地 | 后续方向 |
| --- | --- | --- | --- |
| SoccerMaster | 一个足球专用共享模型同时服务检测、跟踪、标定、事件等任务 | 建立统一 `AnalysisContext`、`Observation` 和报告数据契约，各分析器共享同一比赛状态 | 模型公开稳定后增加统一特征适配器，逐步减少重复专家模型 |
| PathCRF | 用动态球员图和时序约束恢复逻辑一致的球权路径 | `event_timeline` 使用带确认、释放约束的球权解码，抑制单帧持球人跳变 | 数据充足后替换为可训练的 Set Attention + CRF/Viterbi 解码器 |
| TacticAI | 用几何深度学习描述球员关系，预测角球接球人和射门，并生成站位建议 | `relational_space` 把球员作为节点，计算球周人数、队间距离和角球禁区结构 | 积累定位球标签后训练接球人、射门概率和站位调整模型 |
| Monte Carlo Pass Search | 用世界模型滚动多个传球方案，并比较执行价值与备选价值 | 保留反事实分析层的位置，但基础版不输出缺少模型支撑的“最佳传球” | 获得稳定球轨迹、球权和价值标签后加入候选传球采样与结果分布 |
| BASKET | 长视频细粒度技能判断需要长期证据，单片段或单帧很不稳定 | 阵型和空间指标跨帧聚合，报告记录有效样本数与代表帧 | 个人技术改为多次射门/传球片段聚合后再做球员级评分 |
| Synthetic Data | 合成数据用途由生成过程决定，必须报告来源、适用范围和效用限制 | 报告固定包含数据来源、算法来源和限制；当前不混入未经验证的合成结果 | 自建隐私保护数据时同时保存生成配置、全局效用和任务专项效用 |
| Skor-xG | 用三维骨骼和球员关系图补充传统射门位置特征 | 当前远景模块先加入防守距离、封堵人数、门将深度和射门线距离，输出 `xg_context_proxy` | 获得射门结果、姿态和三维标签后再训练并校准 GATv2，不把代理值称为正式 xG |
| Passing Network | 用有向图描述传球连接、核心节点和组织分散程度 | 聚合视觉传球候选并计算中心性、最大连通分量和传球熵 | 长视频上加入时间窗动态图和比赛阶段条件，避免整场单一网络掩盖战术变化 |

论文入口：

- [SoccerMaster](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_SoccerMaster_A_Vision_Foundation_Model_for_Soccer_Understanding_CVPR_2026_paper.html)
- [PathCRF](https://arxiv.org/abs/2602.12080)
- [TacticAI](https://www.nature.com/articles/s41467-024-45965-x)
- [Monte Carlo Pass Search](https://arxiv.org/abs/2606.11120)
- [BASKET](https://arxiv.org/abs/2503.20781)
- [Synthetic Data for Sharing and Exploration in High-Performance Sport](https://doi.org/10.1007/s40279-025-02221-6)

## 3. 代码结构

```text
tactical_analysis/
├── models.py                 # 全模块共享的数据契约、证据、结论和事件
├── gsr_io.py                 # SoccerNetGSR JSON 输入适配器
├── features.py               # 共享速度、加速度、方向等视觉轨迹运动学
├── base.py                   # 分析器接口与注册表
├── pipeline.py               # 按优先级调度分析器并隔离模块错误
├── reporting.py              # 报告、比赛摘要、事件时间轴和高光清单输出
└── analyzers/
    ├── quality.py            # P0 数据质量门控
    ├── events.py             # P0 受约束球权路径和事件候选
    ├── spatial.py            # P0 区域占用与队形尺度
    ├── team_shape.py         # P1 持球/无球 Team Shape 与三线结构
    ├── formation.py          # P1 跨帧阵型倾向
    ├── relations.py          # P1 动态球员关系图
    ├── set_pieces.py         # P1 定位球开出、落点与控制
    ├── goalkeeper.py         # P1 射门关联的门将干预候选
    ├── phases.py             # P1 比赛阶段与防守高度分段
    ├── progression.py        # P2 传球类型、推进、Packing 与穿线
    ├── attacking_chains.py   # P2 射门前可见进攻链路
    ├── defensive_risk.py     # P2 防守三区危险来源与通道
    ├── numerical_superiority.py # P3 持续动态多打少及后续结果
    ├── defensive_gaps.py     # P3 三线间距与低密度防守空档
    ├── transitions.py        # P3 五秒转换速度、质量与阵型恢复
    ├── pressing.py           # P3 高位逼抢、方向、PPDA 与结果代理
    └── pass_network.py       # P2 传球网络、中心性与传球熵
```

球队身份底座位于框架目录之外的推理层：

- `jersey_color.py`：只截取上半身核心区域，使用中心加权 HSV 特征，并抑制检测框边缘的草地颜色。
- `team_identity.py`：先按时间间隔和持续颜色变化把跨镜头复用 ID 切成身份一致的短轨迹，再对多帧颜色证据加权投票。
- `tracklet_appearance.py`：使用足球域 CLIP 的 `color_embedding` 聚合短轨迹外观；从同场高置信蓝/白球员建立原型，只复核低置信轨迹并补全待确认身份。
- `refine_team_identities.py`：串联逐帧球衣估计、短轨迹切分、颜色投票、CLIP 原型复核和每帧最多 3 名裁判约束。
- `team_identity_<video>.json`：记录每条轨迹的身份、置信度、证据帧数和颜色得分，便于定位误判。

当蓝/白等颜色证据达不到置信度与差值门槛时，系统使用同场 CLIP 原型进行二次判断；仍无充分证据的轨迹保留为橙色“待确认”，不再根据当时位于球场左侧或右侧强行归队。配置了球衣先验时，JSON 内部固定 `team0/left=蓝队`、`team1/right=白队`，门将和“误判为球后恢复成人物”的分支也使用同一映射，不按半场位置翻转球队。

对于远景小目标，推荐两级运行策略：预览使用快速颜色特征与短轨迹投票；正式分析关闭 `--fast-mode`，启用 CLIP/ReID。CLIP 原型复核应设置颜色置信度上限与较高特征差异门槛，不应无条件覆盖明确的用户颜色先验。后续可将 SigLIP 作为高精度外观编码器接在同一接口下，而不改动下游战术模块。

## 3.1 主流技术栈与本项目选择

| 能力 | 常见方案 | 本项目当前方案 | 决策 |
| --- | --- | --- | --- |
| 球员检测 | YOLOv5/v8、YOLOX | SoccerNetGSR 的足球域 YOLOX | 暂不因通用模型更新而替换；先以球员召回和下游战术指标准确率做 A/B 测试 |
| 跨帧追踪 | ByteTrack、DeepSORT | Deep-EIoU、ReID、IDATR tracklet refinement | 保留基线优势，同时让所有分析器只依赖统一 `track_id` 契约 |
| 球队分类 | K-Means、SigLIP、CLIP | CLIP、球衣颜色聚类与用户颜色先验 | 明确支持赛前球衣颜色输入；后续用 SigLIP 做遮挡和光照变化下的视觉表征对照 |
| xG | 逻辑回归、GBDT、神经网络、Skor-xG/GATv2 | `xg_lite` 几何基线 + `xg_context_proxy` 上下文排序 | 没有射门结果训练集前不输出“正式 xG”；代理值只用于候选排序和可解释复核 |
| 战术关系 | 几何图、GNN、时空 Transformer | 动态几何关系图、Packing、传球网络 | 先固定图数据契约和可解释基线，再用公开 tracking 数据训练 GNN 替换评分函数 |
| 战术生成 | TacticAI 类几何生成模型 | 只描述角球站位，不自动生成建议 | 需要定位球结果标签与专家审核闭环后再进入生成阶段 |

本项目明确排除 GPS、GNSS、IMU、心率、训练负荷和伤病预测。即使数据文档包含这些方向，当前流水线只消费转播/固定机位视频及其视觉派生轨迹。

## 3.2 三层数据契约

1. **事实层**：帧、时间、球员 ID、球队、角色、球场 XY、速度、加速度、方向、球 XY 与候选球权。
2. **描述层**：事件时间轴、四阶段比赛状态、Team Shape、阵型倾向、推进、Packing、压迫和传球网络。
3. **解释层**：压迫效率、转换质量、阵型恢复、机会质量、无球空间创造和战术模式。现阶段只输出有证据的候选，不生成教练级确定结论。

现有视频脚本保持独立并继续使用：

- `make_tactical_visualization_video.py`：远景战术可视化；右上角简洁战术板只保留标准球场、蓝白两队、裁判和足球点位，不绘制粒子、热力、拖尾、威胁圆或球队外接框。
- `make_tactical_report_video.py`：按证据原时间穿插中文暂停分析，暂停页使用干净原始帧，结尾继续保留总结；阵型章节直接在真实代表帧绘制站位线。
- `make_formation_analysis_video.py`：阵型线与关键帧暂停。
- `make_offensive_analysis_video.py`：射门、传球、推进和定位球代理指标。
- `make_individual_technique_analysis_video.py`：近景射门和门将姿态分析。
- `run_match_analysis_workflow.py`：完整比赛远近景扫描与片段路由。

## 4. 运行方法

先从视频生成 GSR JSON，再运行统一分析框架：

```bash
conda activate sports

python run_tactical_analysis.py \
  --json-path soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-1000/SNGS-1000.json \
  --input-video soccer_input_dataset/test_5min.mp4 \
  --output-dir soccer_input_dataset/outputs/tactical_framework_test_5min
```

只运行指定模块：

```bash
python run_tactical_analysis.py \
  --json-path /path/to/result.json \
  --fps 25 \
  --analyzers data_quality,event_timeline,spatial_structure
```

新版纯视觉优先链为：

```text
data_quality -> event_timeline -> spatial_structure -> team_shape_engine
             -> formation_tendency -> relational_space -> phase_of_play
             -> set_piece_delivery -> goalkeeper_interventions
             -> progression_analysis -> attacking_chains -> defensive_third_risk
             -> pass_network -> numerical_superiority -> pressing_analysis
             -> defensive_gaps -> transition_analysis
```

已知球衣颜色时应显式传入，而不是让系统从整段视频的最高频颜色猜测球队：

```bash
python run_local_video_gsr_visualization.py \
  --input-video /path/to/match.mp4 \
  --team0-colors blue \
  --team1-colors white,red \
  --referee-colors red \
  --goalkeeper-team0-colors yellowgreen \
  --fast-mode --recall-optimized
```

如果一队球衣本身是白红条纹，应同时传入 `white,red`。裁判也为红色时存在颜色提示冲突，此时应提供更准确的完整球衣描述并使用非快速 CLIP 模式；重叠颜色不会在轨迹投票中同时给两个身份加票。

查看可用分析器：

```bash
python run_tactical_analysis.py --list-analyzers
```

输出包括：

- `tactical_analysis_report.json`：前端和后续模型可直接读取的完整结构化结果。
- `tactical_analysis_report.md`：中文结论、置信度和限制说明。
- `event_timeline.csv`：可用于视频裁剪和人工复核的候选时间轴。
- `match_summary.json`：比赛级事件、球权、Team Shape、推进、压迫、转换和球员贡献摘要。
- `highlight_manifest.csv`：按技战术价值排序、带前后时间窗的高光裁剪清单。

按高光清单导出片段：

```bash
python export_tactical_highlights.py \
  --input-video /path/to/match.mp4 \
  --manifest /path/to/highlight_manifest.csv \
  --output-dir /path/to/highlight_clips \
  --top-n 10
```

生成按原时间穿插、并保留结尾总结的视频：

```bash
python make_tactical_report_video.py \
  --input-video /path/to/tactical_base.mp4 \
  --report-json /path/to/tactical_analysis_report.json \
  --output-video /path/to/tactical_review.mp4 \
  --interleave-chapters
```

新版默认暂停时长已经减半：标题 2.5 秒、每个分析章节 5 秒、结尾总结 5 秒；阵型关键帧默认 0.6 秒。仍可通过 `--title-seconds`、`--chapter-seconds` 和 `--summary-seconds` 调整。

阵型跨时间投票达到门槛时输出阵型倾向；未达到门槛但存在足够完整远景时，只输出“低置信阵型代表帧”，在该帧连接可见层次并注明全片支持率，不把瞬时结构当作整场固定阵型。

## 5. 扩展新分析器

2026-10-06 更新：网页已增加按需阵型框线图、原图对照和标注短片；人数不全时保留实际可见分线，不再折算十人。定位球连续控制、门将事件时间关联、复核统计和组织链镜头约束也已修正。入口、测试和限制详见 [阵型可视化与分层分析优化](阵型可视化与分层分析优化_2026-10-06.md)。

新增分析器只需继承 `TacticalAnalyzer`、注册名称并返回 `AnalyzerOutput`。分析结论必须带来源帧、时间、置信度和限制，随后把名称加入 `configs/tactical_analysis.yaml` 的优先级链。

推荐下一轮按以下顺序扩展：

1. P1 在 SoccerNet GSR、SoccerTrack v2、SkillCorner Open Data、Metrica 上建立事件链和 Team Shape 的下游评测，不只看检测/跟踪 HOTA。
2. P2 把几何规则生成的关系图样本保存成训练格式，加入时序 GNN 做比赛阶段、穿线和压迫模式判别对照。
3. P2 使用公开事件数据和 `socceraction` 验证 SPADL、xT、VAEP，并完成概率校准后再升级为正式价值指标。
4. P3 为角球建立事件窗口与关系图样本，先做相似战术检索，再做接球人预测和站位建议。
5. P3 使用姿态窗口和多次动作聚合建立个人射门、头球、门将扑救的技能档案。
6. P4 训练可见性/遮挡模型和时空插补，满足完整 22 人轨迹后再实现可靠的 Pitch Control、无球跑动和防守责任归因。
7. P4 在公开 3D 球轨迹数据上验证 Skor-xG 或反事实传球评估，成熟后再接入本项目。
