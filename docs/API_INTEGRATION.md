# 外接视觉模型 API 调用说明

更新：2026-10-09。以下描述代码当前行为，不保证任何模型名称在你的账户中可用。本次发布不调用付费模型。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `run_openai_tactical_review.py` | 事件审核、分段扫描、边界补审、组织/空间复核与网页报告 |
| `tactical_analysis/openai_review.py` | Responses 请求、base64 图片、结构化约束、SSE 解析、重试、使用记录、响应校验 |
| `review_tactical_candidates_with_vision.py` | 旧 Chat Completions 单事件实验接口，抽帧/门将裁剪/证据保存也供新版复用 |
| `build_layered_tactical_review.py` | 默认无 API 的分层候选包，`--review-api` 启用旧路径 |
| `prepare_assistant_review_sheets.py` / `prepare_layered_review_evidence.py` | 准备直接看图审核材料，不等于自动完成审核 |
| `build_assistant_review_version.py` / `build_reviewed_tactical_layers.py` | 将已有复核记录整理为网页版本 |
| `web_preview/server/local-analysis.js` | 根据后端环境变量选择入口并运行 Python |

`API/football_inference/api.py` 是 RF-DETR 足球检测接口，与外部大模型调用不是一回事。

## 证据与审核

新版候选前后通常各取 5 秒，基础 21 张时序图；必要时扩展窗口、加密采样或裁剪。门将需身份、来球和应对动作证据。出界与重启类型分别判断，不能用出界直接确认角球。模型输出经 JSON schema、时间和帧索引校验，得到支持、修正、否定或不确定，再转换成观察、解释、建议与限制。

保留证据、失败状态、缓存指纹、`api_usage.json`；调用失败不得输出假确定结论。一个节点可能多次请求，节点数不是账单调用数。

官方 Responses API 支持文本和图片输入，具体模型能力另行核对：[官方接口概览](https://developers.openai.com/api/reference/responses/overview)。本项目提交抽帧图片，不是直接向模型发送整个 MP4。

## 密钥

只放后端环境变量，不使用 `VITE_` 前缀，不写进源码、网页或 Git。Bash 可隐藏输入：

```bash
read -rsp 'API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
```

也可填写被忽略的 `.env` 并显式加载，仅分享 `.env.example`。曾出现在聊天或日志中的密钥应在提供方后台撤销并更换；删除文本不会撤销凭证。

## 官方协议路径

现有默认常量 `gpt-5.6-sol` 用于保留历史实验，不保证账号可用性、价格或效果。当前只内置此名字的历史估算计价，未实时同步价格，不是严格账单上限。

```bash
python run_openai_tactical_review.py \
  --video soccer_input_dataset/input_match.mp4 \
  --report-json soccer_input_dataset/outputs/local-match/report/tactical_analysis_report.json \
  --output-dir soccer_input_dataset/outputs/local-match/visual_review \
  --prompt-key --max-events 3 --max-calls 4 --max-usd 2
```

先确认模型权限和图片能力。`--prompt-key` 仅当前 CLI 读取，不给网页服务设置密钥。脚本还会做分段扫描，因此 `--max-events 3` 不等于最多三个任务；`--max-calls` 限制累计请求尝试次数，重试也计入。

其他实际可用模型用 `--model MODEL_ID`。未知计价时当前代码要求显式 `--no-cost-limit`，仍需保留小的 `--max-calls` 并在平台侧设预算。未适配测试前不能承诺 reasoning、JSON schema、响应格式全部兼容。

## 第三方兼容网关

此入口需要 Responses/SSE 兼容性，不适用于只支持 `/chat/completions` 的平台：

```bash
read -rsp 'Relay key: ' RELAY_API_KEY
export RELAY_API_KEY
export RELAY_BASE_URL='https://your-provider.example/v1'
export RELAY_MODEL='replace-with-your-available-vision-model'

python run_openai_tactical_review.py \
  --video soccer_input_dataset/input_match.mp4 \
  --report-json soccer_input_dataset/outputs/local-match/report/tactical_analysis_report.json \
  --output-dir soccer_input_dataset/outputs/local-match/relay_review \
  --base-url "$RELAY_BASE_URL" --model "$RELAY_MODEL" \
  --api-key-env RELAY_API_KEY --reasoning-effort medium \
  --response-format schema --no-cost-limit --max-calls 4 --max-events 3
```

脚本构造 `/v1/responses`。若网关不支持 schema 可试 `--response-format json_prompt`，本地仍校验 JSON；不保证任意网关可用。第三方名称可能是别名，无法凭字符串验证实际模型身份。

`--no-cost-limit` 取消金额限制，但请求次数限制仍在。费用看平台账单。视频帧会发送给服务商，须先确认数据授权和隐私，不得把官方密钥发送给第三方域名。

## 网页启用优先级

在启动 Vite 的同一终端设置变量后重启服务：

1. `RELAY_API_KEY` + `RELAY_BASE_URL`：兼容网关；可设 `RELAY_MODEL`、`RELAY_REASONING_EFFORT`、`RELAY_RESPONSE_FORMAT`。当前网页固定最多 150 次且不设金额上限，建议先用上述 4 次 CLI 验证。
2. 否则 `OPENAI_API_KEY`：当前网页固定默认模型，最多 80 次、默认 8 美元历史估算；`OPENAI_REVIEW_MAX_USD` 可调。其他模型使用 CLI 或明确修改后端。
3. 否则 `DEEPSEEK_API_KEY`：旧 Chat Completions 实验路径。必须核对端点和模型支持图片，文字响应成功不代表模型看到了图片。
4. 都为空：仅生成本地候选与证据，不调用外部模型。

同一五分钟样例可能命中校验过的缓存，不一定产生新账单；检查本次调用次数与任务日志。

## 离线与边界

- `--cached-only` 只从完整有效缓存重新发布，缺失即报错，不发请求。
- 旧单事件脚本 `--dry-run` 只准备和检查证据。
- 单元测试使用构造响应/模拟，不需要真实密钥，不验证提供方当前在线可用性。
- 历史“助手直接复核”不是后台常驻服务。新上传不会自动复现助手看图过程；需可用视觉接口或另行复核标注。
- 无 API 可做检测、规则候选、几何统计与证据浏览；有 API 也不保证与人工专家或过去样例相同效果。
