<p align="center">
  <img src="./assets/fastjev-logo.svg"
       width="420"
       alt="FastJev" />
</p>

<div align="center">

[English](README.md) | 简体中文

<img src="assets/fastjev-cover.webp" alt="FastJev 自托管语义决策 SDK" width="100%">

**使用开放模型快速完成自托管语义决策。**

</div>

FastJev 是面向自托管语义路由、LLM 分类和结构化 AI 决策的开源 Python SDK。输入非结构化状态、运行时定义的问题和类型化选项，即可直接获得选项概率，无需生成回答文本或解析 JSON。

它适合 AI agent 和应用中的路由、重试策略、证据判断等范围明确的决策。

## 为什么选择 FastJev

- **无需生成文本：** 直接读取选项 logits，不执行解码循环，也无需修复生成结果。
- **类型化 Python API：** 提供 `Choice`、`Boolean` 和 `Score`，并返回稳定选项 ID 与来源元数据。
- **开放模型自托管：** 支持 PyTorch/CUDA、vLLM 和 Apple Silicon 上的 MLX。
- **面向重复决策：** 模型常驻内存，支持批量评分和共享状态前缀复用。
- **结果可审计：** 固定模型 revision，并公开 prompt、逐行输出、基准 runner 和校验和。
- **兼容 API：** 可选 HTTP 服务支持公开文档中的 System One 请求格式。

## 快速开始

默认后端需要 Python 3.10+、CUDA，以及一块能够容纳 4B BF16 模型的 GPU：

```bash
python -m venv .venv
. .venv/bin/activate
export HF_HOME=/path/to/large-drive/huggingface
pip install -e '.[torch]'
```

首次调用会自动从 Hugging Face 下载固定 revision 并缓存到 `HF_HOME`，无需提前手动下载模型。

```python
from fastjev import Choice, FastJev, Option

with FastJev.from_pretrained(
    "Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
) as jev:
    result = jev.decide(
        state="Customer cannot access an account after a password reset.",
        question=Choice("Which queue should handle this request?", [
            Option("access", "Account access support."),
            Option("billing", "Billing support."),
        ]),
    )

print(result.value)
print(result.probabilities)
```

远程模型必须使用不可变的 40 字符 Hugging Face revision。本地模型目录也受支持，但需要提供非空 revision 标签来记录结果来源。

使用本地或 Hugging Face 托管的 GGUF 文件时安装 `.[llama-cpp]`。[Python SDK 指南](docs/SDK.zh-CN.md)介绍 llama.cpp 的设置与来源记录。

## 后端

| Runtime | 安装 | 适用场景 |
|---|---|---|
| PyTorch/CUDA | `pip install -e '.[torch]'` | 默认直接 logits 评分 |
| vLLM/CUDA | `pip install -e '.[vllm]'` | 批量常驻推理 |
| MLX/Apple Silicon | 查看 [MLX 指南](docs/MLX.zh-CN.md) | 原生 macOS arm64 推理 |

所有后端都实现同一个 `ScoringBackend` 协议，因此切换 runtime 时可以保留相同的类型化决策和结果对象。批处理、后端注入、结果语义与生命周期管理见 [Python SDK 指南](docs/SDK.zh-CN.md)。

如需纯浏览器本地推理，可使用 [WebGPU demo](webgpu-demo/index.html)；它按需下载固定的 GGUF 模型，并使用浏览器缓存。

## CLI 与 HTTP API

通过命令行评分 JSONL：

```bash
CUDA_VISIBLE_DEVICES=0 fastjev-score \
  --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl \
  --output results.jsonl
```

当所有输入行具有相同状态时，可用 `--mode shared` 让多项标准复用一个前缀。如需常驻服务，请安装 `.[api]` 并运行 `fastjev-serve`；[HTTP API 指南](docs/SYSTEM_ONE_API.zh-CN.md)说明了 `POST /v1/systemone`、认证方式和兼容边界。

## 实测性能

在一块 RTX 3090 上，固定的 Qwen3.5-4B 基准使用相同状态评估 21 项二元标准：

| 输出路径 | 中位耗时 | 输出 token |
|---|---:|---:|
| 直接类型化 logits | **1.023 秒** | **0** |
| 自回归 JSON 数组 | 5.332 秒 | 111 |

生成基线只输出有序的 `"yes"`/`"no"` 数组，完整输出耗时是直接读取的 5.21 倍；其选择与直接 argmax 在 21 项标准中的 18 项一致。这项实验只测量输出路径成本，不表示两种读取方式语义等价。查看[原始运行记录](results/raw/decision-vs-compact-array.json)以及完整的[结果与限制](docs/RESULTS.zh-CN.md)。

## 文档

- [Python SDK](docs/SDK.zh-CN.md) — 类型化决策、批处理和后端协议
- [System One 兼容 API](docs/SYSTEM_ONE_API.zh-CN.md) — 服务配置和 wire format
- [MLX 后端](docs/MLX.zh-CN.md) — Apple Silicon 配置与缓存行为
- [结果](docs/RESULTS.zh-CN.md) — 速度、质量、扰动测试和声明边界
- [方法](docs/METHOD.zh-CN.md) — 固定 prompt、指标和计时范围
- [复现](docs/REPRODUCE.zh-CN.md) — 固定环境和验证命令
- [基准包](benchmarks/README.zh-CN.md) — fixture、runner 和来源选择
- [交互回放](demo/index.html)与[纯浏览器 WebGPU demo](webgpu-demo/index.html)

## 限制与来源

FastJev 返回的是以所给选项为条件的概率，并非经过校准的置信度。请在实际决策工作负载上完成验证和校准。快速前缀复用路径也可能改变 BF16 argmax，目前仍属实验功能。

本仓库不分发模型权重或第三方评估记录。上游模型沿用各自许可证，精确 revision 记录在 [THIRD_PARTY.zh-CN.md](THIRD_PARTY.zh-CN.md)。

FastJev 是 [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf)（原名 OpenJev）的独立维护分支，保留原始 Git 历史和 MIT 许可证，但采用独立路线图。FastJev 与 TheoLeeCJ、TypeSafe 或 Jev 无隶属关系，也未获得其背书；它不复现 Jev 未公开的模型、训练方法、校准能力或性能。

项目代码按 [MIT 许可证](LICENSE)发布。
