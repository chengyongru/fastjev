# System One 兼容 API

[English](SYSTEM_ONE_API.md) | 简体中文

fastjev 可以公开 TypeSafe 文档中 System One HTTP API 的 wire-compatible 子集。适配器在 `POST /v1/systemone` 接受相同的顶层 `state`、`model` 和 `questions` 字段，并返回 `model`、`answers` 和 `usage`。这是独立兼容层：它不运行 Jev，fastjev 分数也没有按 Jev 概率校准。

实现的约定遵循公开的 [TypeSafe API reference](https://docs.typesafe.ai/api)：

- `noul` 映射为两个 fastjev 选项，并返回分配给 `true` 的概率。
- `choice` 返回概率最高的选项以及完整选项分布。
- `score` 将有序标准视为 `0..N-1` 各级，并返回按概率加权的数值。
- 多个问题共享同一个请求状态，并通过 fastjev 的直接选项 logits 路径独立评分。本服务不公开实验性的共享前缀评分器，因为仓库未声明它的决策与直接评分语义等价。
- `GET /v1/models` 返回唯一配置的 fastjev 模型。

## 安装与运行

在 fastjev 所在的同一隔离环境中安装 API extra：

```bash
pip install -e '.[api,torch]'
```

本地无认证服务保留默认 loopback 绑定：

```bash
CUDA_VISIBLE_DEVICES=0 fastjev-serve \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --served-model fastjev-qwen3.5-4b \
  --served-model-description 'fastjev direct option-logit baseline on Qwen3.5-4B BF16' \
  --served-model-release-date 2026-09-18
```

绑定非 loopback 地址时，通过命名环境变量设置 bearer token。除非显式传入 `--allow-unauthenticated`，服务器会拒绝未认证的非 loopback 绑定。

```bash
export FASTJEV_API_KEY='replace-with-a-secret'
CUDA_VISIBLE_DEVICES=0 fastjev-serve \
  --host 0.0.0.0 \
  --model /path/to/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --served-model fastjev-qwen3.5-4b \
  --served-model-description 'fastjev direct option-logit baseline on Qwen3.5-4B BF16' \
  --served-model-release-date 2026-09-18
```

进程刻意只运行一个 Uvicorn worker。增加 worker 会让每个 worker 各自加载一份模型。请求围绕常驻模型串行执行，避免并发 HTTP handler 争用同一个 GPU 模型。

## 请求

```json
{
  "model": "fastjev-qwen3.5-4b",
  "state": "Help! My payouts have been failing for three days.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?"
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, and refunds",
        "technical": "Bugs, outages, and integrations"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the problem?",
      "criteria": ["Minor", "Degraded", "Blocking"]
    }
  }
}
```

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @request.json
```

配置认证后，还需发送 `Authorization: Bearer $FASTJEV_API_KEY`。

## 兼容边界

适配器保留公开 wire shape，不复现 Jev 的模型行为：

- 请求中的 `model` 必须等于配置的 fastjev 模型 ID。`jev-latest` 等 Jev 名称或别名会被拒绝，不会被冒充。
- fastjev 当前为 `choice` 支持 2–16 个选项；TypeSafe 文档上限为 255。
- `score` 支持文档规定的 2–10 级。`noul` 和 `choice` 的标准可以使用字符串、JSON 对象、数组或 `null`；结构化内容会以 JSON 形式渲染到 fastjev prompt。
- TypeSafe SDK 允许省略 instructions 或设为 `null`。适配器会按类型提供通用问题，但显式 instructions 仍然更合适，因为它定义了预期决策边界。
- fastjev 默认对每个转换后问题限制 4,096 个输入 token，且不做截断。TypeSafe 文档使用不同的上下文预算。
- Choice 和 Score 的文档响应要求包含 `confidence`。TypeSafe 没有公开其精确统计量，因此适配器返回 `1 - normalized entropy`，并在顶层 `fastjev` extension 中标为 `one-minus-normalized-entropy`。该数值不能与 TypeSafe confidence 直接比较。
- `usage.input_tokens` 是各独立问题 prompt 长度的总和。`usage.output_tokens` 为零，因为 fastjev 读取选项 logits，不生成回答文本。
- 返回的 `fastjev.probability_status` 会记录选项概率具有条件性且未校准。在对重要操作进行自动化前，请在部署工作负载上验证阈值。

FastAPI 还在 `/docs` 和 `/openapi.json` 提供自动生成的 OpenAPI 文档。这些便利路由是 fastjev extension，不是 TypeSafe 端点。
