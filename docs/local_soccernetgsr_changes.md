# Local SoccerNetGSR Changes

本文档记录当前本地版本相较于基础克隆仓库新增或调整的内容，便于后续继续实验时保留当前可用版本。

## 当前保留版本

- 输入视频：`soccer_input_dataset/test.mp4`
- 当前最终推荐输出视频：`soccer_input_dataset/outputs/test_gsr_visualized_full_referee_magenta.mp4`
- 当前最终推荐 JSON：`soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-999/SNGS-999.referee_refined.json`
- 当前稳定基础 JSON：`soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-999/SNGS-999.json`
- 当前效果：完整 `744` 帧，`25fps`，`29.76s`，`1280x720`；人物召回较高，足球只保留稳定单轨迹，裁判从两队中分离并使用独立颜色显示。

## 相对基线仓库的优化总览

基础克隆仓库主要面向 SoccerNetGS 数据目录运行。当前本地版本围绕 `test.mp4` 增加了完整本地视频输入流程、球员召回优化、足球误检过滤、裁判后处理和更清晰的可视化输出。

完整流程图见：`docs/soccernetgsr_pipeline_flowchart.md`

整体优化链路如下：

1. 本地 mp4 输入适配：将普通视频抽帧并组织为 `SoccerNetGS/test/SNGS-999/img1` 格式。
2. 完整推理流程串联：自动执行关键点、检测、跟踪、tracklet refinement、pitch 坐标转换、JSON 写出和视频合成。
3. 球员召回提升：通过低阈值检测/跟踪配置减少漏检。
4. 足球误检抑制：通过几何约束、Ball 投票比例、持续帧数和 Lucas-Kanade 光流一致性过滤足球候选。
5. 裁判识别优化：基于第三服装颜色、原始 Referee 投票和轨迹长度，把裁判从两队球员中分离出来。
6. 可视化优化：足球黄色，裁判洋红色，左右两队红/蓝色；右上角战术图同步使用同样的颜色区分。

## 新增文件

### `run_local_video_gsr_visualization.py`

本地视频端到端运行脚本，用于把普通 `.mp4` 输入转换成 SoccerNetGSR 项目所需的数据目录，并串联原项目主要流程：

- 视频抽帧到 `img1/`
- 关键点和单应性估计
- GSR 检测、角色/号码/队伍推理
- IDATR tracklet 生成与 refine
- 生成 pitch 坐标文件
- 写出 SoccerNetGS 格式 JSON
- 生成可视化帧并合成完整 mp4

新增 `--recall-optimized` 模式，用于降低检测/跟踪阈值、提高球员召回。

### `visualize_local_result.py`

本地单视频可视化包装脚本。它调用原项目的 `visualize_prediction_results.visualize_predictions`，但只针对一个本地视频目录运行，避免每次都手动构造路径。

### `refine_referee_roles.py`

独立的裁判后处理脚本。后续裁判优化从这个文件开始，不再直接改当前稳定脚本。

用途：

- 读取当前稳定 JSON 和 `refined_SNGS-999.txt`
- 根据第三服装颜色、原始 Referee 投票、轨迹长度等规则识别裁判轨迹
- 输出新的 `*.referee_refined.json`
- 输出裁判识别报告 `*.referee_refined_report.md`
- 可选生成单独的裁判修正版可视化视频

当前 `test.mp4` 上自动识别出的裁判轨迹：

- `track_id=4`：主裁，原先被分到 `right` 队，主要颜色为 `red`
- `track_id=40`：另一名裁判候选，原先被分到 `right` 队，主要颜色为 `orange/red/yellow`

对应输出：

- `soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-999/SNGS-999.referee_refined.json`
- `soccer_input_dataset/gsr_demo/SoccerNetGS/test/SNGS-999/SNGS-999.referee_refined_report.md`
- `soccer_input_dataset/outputs/test_gsr_visualized_full_referee_refined.mp4`
- `soccer_input_dataset/outputs/test_gsr_visualized_full_referee_magenta.mp4`

## 修改过的原项目文件

### `write_json_file_team.py`

主要新增足球后处理逻辑：

- 读取 `POSTPROCESS` 配置
- 足球必须满足面积、宽高比、宽/高上限、Ball 投票比例、连续帧数
- 加入 Lucas-Kanade 光流一致性检查，避免静态噪声或人物局部小框被误标成足球
- 被拒绝的 Ball 候选会回退为最可信的非 Ball 角色，或标为 `other`
- CLI 只处理 `img1` 目录，避免重复扫描可视化目录

### `visualize_prediction_results.py`

主要新增可视化配置、足球显示优化和裁判显示优化：

- 从 `soccer_input_dataset/gsr_demo/config.local.yaml` 读取 `VISUALIZATION`
- 足球使用黄色框和 `Ball` 标签
- 右上角战术图中足球点和人物点同大小，只用颜色区分
- 裁判使用洋红色框和 `Referee` 标签，与红/蓝球队和黄色足球区分
- 右上角战术图中裁判点也使用洋红色

### `soccer_input_dataset/gsr_demo/config.local.yaml`

本地 demo 配置，由本地运行脚本生成并保留。包含：

- 本地 `DATA_DIR`
- 召回优化过的 tracker 参数
- 足球后处理阈值
- 光流筛选阈值
- 可视化点大小和足球框粗细

## 当前关键优化参数

```yaml
POSTPROCESS:
  BALL_MAX_AREA: 500
  BALL_MAX_ASPECT: 1.5
  BALL_MAX_HEIGHT: 20
  BALL_MAX_WIDTH: 20
  BALL_MIN_FRAMES: 10
  BALL_MIN_VOTE_RATIO: 0.6
  BALL_USE_OPTICAL_FLOW: true
  BALL_MIN_FLOW_VALID_RATIO: 0.7
  BALL_FLOW_MAX_PAIRS: 120
  BALL_FLOW_MAX_RESIDUAL: 6.0
  BALL_FLOW_MAX_ERROR: 30.0

VISUALIZATION:
  PLAYER_RADAR_RADIUS: 2
  BALL_RADAR_RADIUS: 2
  BALL_BOX_THICKNESS: 3
```

## 裁判后处理参数

`refine_referee_roles.py` 的默认参数如下：

```bash
python refine_referee_roles.py \
  --official-colors red,orange,yellow,neon\ yellow \
  --min-frames 30 \
  --min-referee-votes 8 \
  --min-referee-ratio 0.10 \
  --min-official-color-ratio 0.60
```

判断思路：

- 只处理最终 JSON 中仍为 `player` 的轨迹，避免影响足球、守门员等类别
- 利用 `refined_SNGS-999.txt` 中的逐帧原始角色投票和颜色投票
- 如果轨迹服装颜色高度集中在裁判色系，并且存在一定数量的 Referee 原始投票，则改为 `referee`
- 改为裁判后清空 `jersey`、`team`、`color`，避免继续被左右队统计使用

当前裁判修正后统计：

```text
player: 9008
referee: 829
ball: 444
goalkeeper: 166
```

## 已生成的重要输出

- `soccer_input_dataset/outputs/test_gsr_visualized_full.mp4`：完整基础输出
- `soccer_input_dataset/outputs/test_gsr_visualized_full_balanced.mp4`：球员召回和足球过滤均衡版
- `soccer_input_dataset/outputs/test_gsr_visualized_full_ball_refined.mp4`：更严格足球几何过滤版
- `soccer_input_dataset/outputs/test_gsr_visualized_full_ball_flow_refined.mp4`：当前保留版，加入光流一致性过滤
- `soccer_input_dataset/outputs/test_gsr_visualized_full_referee_refined.mp4`：裁判从两队中分离后的可视化结果
- `soccer_input_dataset/outputs/test_gsr_visualized_full_referee_magenta.mp4`：当前最终推荐版，裁判使用洋红色框和点显示

## 后续修改约定

- `test_gsr_visualized_full_ball_flow_refined.mp4` 对应的代码和 JSON 作为当前稳定基础版本保留
- 新实验优先新建脚本或输出新文件，避免覆盖已验证结果
- 如果某一轮新实验效果确认更好，再把文档中的“当前最终推荐输出”更新到对应文件
