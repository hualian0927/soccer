# 新版足球技战术分析要求实现对照

## 1. 范围与原则

本对照以 `足球技战术分析调研.pdf` 的新版路线为准，只处理视频视觉信息，不接入 GPS、IMU、生理或训练负荷数据。系统沿“视频事实 -> 事件 -> 比赛状态 -> 队形与关系 -> 技战术指标 -> 摘要与片段”的链条构建；缺少完整轨迹、官方事件结果或训练标签时，输出必须标记为候选或代理指标。

## 2. 分级实现状态

| 级别 | 新要求 | 当前状态 | 本轮实现或边界 |
| --- | --- | --- | --- |
| P0 | 场地标定、球员/球检测跟踪、球队与角色身份 | 已有底座，持续优化 | 沿用 SoccerNetGSR、短轨迹颜色投票、CLIP 原型复核和裁判人数约束；仍受转播切镜与小球漏检影响 |
| P0 | 触球、传球、接球、带球、射门基础事件 | 已实现候选链 | 新增 `touch_candidate`、`receive_candidate`、`carry_candidate`；射门补充位置、通道和禁区内外上下文 |
| P1 | 球权与比赛四状态 | 已实现代理 | 受约束球权路径输出夺回、持球、丢失、无球四状态及持续时间 |
| P1 | Team Shape | 已实现 | 持球/无球分别计算宽度、纵深、重心、三线高度、线间距和凸包面积 |
| P1 | 阵型与比赛阶段 | 已实现候选 | 跨帧阵型投票；阶段覆盖后场组织、中场推进、前场进攻、长传、定位球和防守高度 |
| P2 | 推进、三区进入、穿线与 Packing | 已实现代理 | 新增肋部进入、转移、传中和直塞候选；Packing 依赖可见球员，不等同完整 tracking 真值 |
| P2 | 压迫位置、方向、触发和效率 | 已实现代理 | 输出高/中/低位、边路/中路方向、触发情景、参与人数、结果与 PPDA 代理；PPDA 不能替代官方事件口径 |
| P2 | 攻防转换和阵型恢复 | 已实现代理 | 输出 5 秒推进、参攻/回追人数、反抢夺回以及丢球后阵型恢复时间 |
| P2 | 传球网络和球员贡献 | 已实现基础版 | 输出连接、中心性、连通性、传球熵及球员事件计数；身份切换会影响跨镜头统计 |
| P2 | 比赛摘要、高光排序与可视化交付 | 已实现 | 新增 `match_summary.json`、`highlight_manifest.csv`、高光 MP4 导出和中文穿插讲解视频 |
| P3 | 动态多打少、高位逼抢/PPDA、防守空档、转换速度 | 已实现候选 | 新增持续人数优势、线间空档和 P3 时间轴事件；压迫与转换升级为 P3，均保留视觉代理边界 |
| P3 | 正式 xG/xT/VAEP、Pitch Control | 部分/待训练 | 当前只有几何和上下文代理；正式概率模型需要带结果标签的数据训练、校准和独立评测 |
| P3 | 无球跑动、盯人、协防、压迫模式识别 | 待增强 | 需要更完整的 22 人时空轨迹、可见性建模和时序 GNN/Transformer，当前不输出确定性责任归因 |
| P4 | TacticAI 类定位球检索、预测与生成 | 研究项 | 需角球接球人、射门结果、站位调整和专家评价闭环，不以启发式规则伪装生成模型 |
| P4 | 3D 姿态、真实射门力量、门将动作评分 | 研究项 | 需要近景姿态、3D 球轨迹和多视角/标定数据；远景二维球速只能作为粗略代理 |

## 3. 本轮代码落点

| 文件 | 作用 |
| --- | --- |
| `tactical_analysis/analyzers/events.py` | 统一基础事件链和射门上下文 |
| `tactical_analysis/analyzers/team_shape.py` | 持球/无球 Team Shape 与三线结构 |
| `tactical_analysis/analyzers/phases.py` | 比赛阶段和四状态持续时间 |
| `tactical_analysis/analyzers/progression.py` | 推进、三区/肋部进入、转移、传中、直塞、Packing 和穿线 |
| `tactical_analysis/analyzers/pressing.py` | 压迫高度、方向、触发、强度、结果和 PPDA 代理 |
| `tactical_analysis/analyzers/transitions.py` | 进攻转换、反抢、回追与阵型恢复 |
| `tactical_analysis/reporting.py` | 完整报告、比赛摘要和高光清单 |
| `export_tactical_highlights.py` | 按清单导出带音频的技战术片段 |
| `make_tactical_report_video.py` | 在证据发生时插入中文分析，并保留结尾总结 |
| `tactical_analysis/analyzers/numerical_superiority.py` | 持续动态多打少、推进和后续结果 |
| `tactical_analysis/analyzers/defensive_gaps.py` | 三线间距、球周密度和通道条件的防守空档 |

## 3.1 优先级总表 P0-P3 对照

| 优先级 | 技战术方向 | 实现模块 | 当前输出 |
| --- | --- | --- | --- |
| P0 | 事件时间轴 | `event_timeline` | 触球、接球、传球、带球、射门、角球和球权转换候选 |
| P0 | 自动高光剪辑 | `reporting.py`、`export_tactical_highlights.py` | 多标签高光清单和 MP4 片段；同一时刻的射门、进攻链与防守风险合并而非重复裁剪 |
| P0 | 射门结构分析 | `event_timeline` | 禁区内外、距离档、射门通道、防守压力、角度及机会质量代理 |
| P0 | 区域统计 | `spatial_structure` | 三区、边中路占用和球队/球员平均站位 |
| P0 | 比赛摘要报告 | `reporting.py` | P0-P3 球队级摘要、完整 JSON、中文 Markdown 和 CSV |
| P1 | 平均站位、阵型、宽度/纵深/线距 | `spatial_structure`、`formation_tendency`、`team_shape_engine` | 跨帧阵型倾向和持球/无球 Team Shape |
| P1 | 定位球落点 | `set_piece_delivery` | 角球/任意球开出、飞行距离、首次稳定落点、区域和落点控制候选 |
| P1 | 门将扑救候选 | `goalkeeper_interventions` | 射门后门将接近、侧向移动、出击和后续控制；异常运动速度会被剔除 |
| P2 | 传球方向、距离与类型 | `progression_analysis` | 前/横/回传，短/中/长距离，短传、长传、直塞、传中互斥分类 |
| P2 | 持球推进 | `event_timeline`、`progression_analysis` | 持续时间、距离、纵向推进和终点区域 |
| P2 | 进攻链路 | `attacking_chains` | 从射门回溯传球、带球、三区进入、穿线与进攻方式 |
| P2 | 防守三区风险 | `defensive_third_risk` | 按对手进入、穿线、传中、推进和射门聚合危险通道 |
| P3 | 动态多打少 | `numerical_superiority` | 连续球周人数优势、推进幅度、前场进入及后续射门 |
| P3 | 高位逼抢 / PPDA | `pressing_analysis` | 压迫高度、参与人数、方向、触发、结果和视觉 PPDA 代理 |
| P3 | 防守空档 | `defensive_gaps` | 三线间距、低防守密度、空档通道及后续威胁 |
| P3 | 攻防转换速度 | `transition_analysis` | 五秒推进速度、质量代理、参攻/回追人数和阵型恢复 |

门将模块存在但不会为了填满报告而强制输出。在测试素材中，所有门将候选都未通过轨迹连续性门槛，因此报告正确输出 0 个候选；这表示“证据不足”，不表示比赛中没有门将动作。

## 4. 输出数据契约

一次分析会形成五个主要文件：

1. `tactical_analysis_report.json`：逐模块结论、证据、置信度、限制和事件时间轴。
2. `tactical_analysis_report.md`：适合人工阅读的中文分析报告。
3. `event_timeline.csv`：可筛选、复核和裁剪的事件表。
4. `match_summary.json`：比赛级球队、球员和阶段摘要。
5. `highlight_manifest.csv`：按分数排序的高光片段及前后时间窗。

## 5. 下一阶段验收顺序

1. 建立 20 至 50 个片段的小型人工真值集，分别评估球队分类、球权、传接球、射门、比赛阶段和阵型，而不是只凭成品视频观感。
2. 在 SoccerNet GSR 与公开完整 tracking 数据上校准阈值，报告 Precision、Recall、F1 和事件时间误差。
3. 训练时序图模型替换穿线、压迫模式和阶段识别中的部分手工规则，并保留当前规则作为可解释基线。
4. 接入正式事件结果标签后训练并校准 xG/xT/VAEP；在此之前保留 `proxy` 命名。
5. 完成可见性和轨迹插补评测后，再开发 Pitch Control、无球跑动与防守责任分析。

## 6. 视频暂停策略

新版默认将旧暂停时间减半：标题 2.5 秒、章节分析 5 秒、结尾总结 5 秒、阵型关键帧 0.6 秒。分析页优先使用事件发生处的干净原帧，阵型章节在真实代表帧上画线；不会删除原视频时间，也保留结尾汇总。
