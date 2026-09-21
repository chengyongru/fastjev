<p align="center">
  <img src="./assets/fastjev-logo.svg"
       width="420"
       alt="FastJev" />
</p>

<div align="center">

[English](README.md) | 简体中文

<img src="assets/fastjev-cover.webp" alt="FastJev 面向自部署的 Jev 开源实现" width="100%">

**在自己的基础设施上部署 Jev 的开源实现。**

</div>

FastJev 是面向自部署的 [Jev](https://docs.typesafe.ai/) 开源实现，也是 [SemIf](https://github.com/TheoLeeCJ/SemIf) 的持续维护分支。固定 revision 的开放模型通过 Torch、vLLM、MLX 和 llama.cpp 运行 `Choice`、`Boolean` 和 `Score` 接口。WebGPU demo 提供浏览器本地推理。

## 为什么选择 FastJev？

FastJev 把 Jev 自部署收敛为标准 Python 工作流。SDK 负责加载模型、验证 2 至 16 个选项、执行评分，并返回包含概率、token 用量、耗时、模型 revision 和 prompt 版本的类型化结果。

常驻 Torch、批量 vLLM、原生 MLX、llama.cpp GGUF、CLI 和可选的 System One 兼容 HTTP API 共享同一结果模型。Torch、MLX 和 llama.cpp 在一次评分中返回结果，输出 token 数量为 0。

每项结果都记录模型 revision 和 prompt 版本。已发布评估还包含逐行数据、原始计时和校验和，用于精确复现。

## 可以直接运行的模型

以下固定 checkpoint 都通过了相同的原生 BF16 直接 logits 接口验证。把任意模型 ID 与 revision 填入后面的快速开始代码即可。

| 模型 | 固定源 revision | 推荐场景 | 自编数据平衡准确率 | 浏览器选项 |
|---|---|---|---:|---:|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | 最小入门模型 | 0.440 | Q8_0，639 MB |
| [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | `12a3808a956f869c767195e9266b59c4d21d92e2` | 体积与质量平衡 | 0.686 | Q4_K_M，1.56 GB |
| **[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)** | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | **推荐模型，实测质量最高** | **0.813** | Q4_K_M，3.01 GB |

表格报告原生 BF16 checkpoint 的质量。浏览器模型采用独立的量化格式。[模型梯度原始报告](results/raw/browser-model-ladder.json)收录精确预测行、revision 和浏览器冒烟结果。

## 快速开始

默认后端需要 Python 3.10+、CUDA 和一块可见 GPU。Qwen3.5-4B 源 checkpoint 占用约 9 GB 磁盘。历史 [RTX 5090 SDK 冒烟测试](https://github.com/chengyongru/fastjev/pull/5)测得峰值 GPU 分配为 7.891 GiB。

```bash
git clone https://github.com/chengyongru/fastjev.git
cd fastjev
python -m venv .venv
. .venv/bin/activate
export HF_HOME=/path/to/large-drive/huggingface
pip install -e '.[torch]'
```

首次调用会自动从 Hugging Face 下载固定 revision 并缓存到 `HF_HOME`。

```python
from fastjev import Choice, FastJev, Option

with FastJev.from_pretrained(
    "Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
) as jev:
    result = jev.decide(
        state={"message": "I was charged twice and need a refund today."},
        question=Choice("Which queue should handle this request?", [
            Option("access", "Account access and authentication."),
            Option("billing", "Billing, payments, and refunds."),
            Option("sales", "Pricing and new contracts."),
        ]),
    )

print(result.value)
print(result.probabilities)
print(result.provenance)
```

远程模型使用不可变的 40 字符 Hugging Face revision。本地模型目录使用描述性 revision 标签记录结果来源。

使用本地或 Hugging Face 托管的 GGUF 文件时安装 `.[llama-cpp]`。[Python SDK 指南](docs/SDK.zh-CN.md)介绍 llama.cpp 的设置与来源记录。

## 实测结果

### RTX 5090 上的 Torch 与 vLLM

同机集成测试使用固定的 Qwen3.5-4B checkpoint、完全相同的三个问题输入（125、152 和 151 token）、一次 warmup 和七次 `decide_many` 计时。

| 后端 | 模型加载 | 三项决策 batch 中位耗时 | 决策/秒 |
|---|---:|---:|---:|
| Torch | **10.04 秒** | 161.97 毫秒 | 18.52 |
| vLLM | 40.43 秒 | **82.73 毫秒** | **36.26** |

在这项 RTX 5090 / WSL 工作负载上，vLLM 吞吐量为 Torch 的 1.96 倍，batch 中位延迟降低 48.9%。额外的 30.38 秒启动成本需要常驻处理约 383 个三项决策 batch 才能抵消。两个后端的三项选择完全一致。

[PR #7](https://github.com/chengyongru/fastjev/pull/7) 记录了这项历史集成测量，包括精确环境和汇总中位数。仓库内的可复现 benchmark bundle 覆盖另外提交了逐行数据的实验。

### 决策质量

| 冻结工作负载 | FastJev 直接 Qwen3.5-4B | Qwen3-Reranker-4B | 已发布 Jev |
|---|---:|---:|---:|
| 自编决策，144 行 | **0.813** | 0.625 | N/A |
| WANLI，256 行 | **0.637** | 0.522 | N/A |
| TypeSafe 公开子集，102 行 / 20 个 case | **0.845** | 0.560 | 0.883 |

前两行使用平衡准确率。第三行使用按 case 等权的众数一致率。Jev 一列复述公开记录。两个开放模型列来自本地冻结评估。[结果文档](docs/RESULTS.zh-CN.md)收录完整数据、扰动测试与声明边界。

## 与相邻项目的区别

这些项目面向相邻的部署需求。各项目发布的性能数据对应各自的工作负载。

| 项目 | 核心机制 | 适合选择它的场景 | FastJev 的区别 |
|---|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | 小型专用决策模型，并行计算选项得分 | 低资源或多语言部署，尤其是愿意按工作负载微调时 | 使用标准开放因果模型处理运行时定义的决策，并提供固定 revision 和 4,096 token 默认输入上限 |
| [semantic-router](https://github.com/aurelio-labs/semantic-router) | 路由样例的 embedding 相似度 | 路由与样例稳定，向量相似度足够解决问题时 | 每次请求都根据完整状态、新标准和选项含义作出判断 |
| [Outlines](https://github.com/dottxt-ai/outlines) | 受约束的自回归生成 | 需要任意 JSON、正则、grammar 或抽取 schema 时 | 专注类型化决策，在一次评分中读取选项分数 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 底层本地推理与 token logprobs | 需要完全控制 runtime 或只做一次性分类器时 | 将 llama.cpp 用作后端，并增加类型化问题、验证、来源记录、多 runtime、HTTP 兼容和冻结评估 |

## 后端与接口

| Runtime | 安装 | 适用场景 |
|---|---|---|
| PyTorch/CUDA | `pip install -e '.[torch]'` | 默认直接 logits 评分 |
| vLLM/CUDA | `pip install -e '.[vllm]'` | 批量常驻服务 |
| MLX/Apple Silicon | 查看 [MLX 指南](docs/MLX.zh-CN.md) | 原生 macOS arm64 推理 |
| llama.cpp/GGUF | `pip install -e '.[llama-cpp]'` | 本地或 Hugging Face 托管的 GGUF 文件 |
| WebGPU/GGUF | 打开[浏览器 demo](webgpu-demo/index.html) | 浏览器本地推理 |

使用 `fastjev-score` 处理 JSONL。安装 `.[api]` 并运行 `fastjev-serve`，即可提供 `POST /v1/systemone` 和 `GET /v1/models`。[SDK 指南](docs/SDK.zh-CN.md)介绍 batching 和自定义后端。[HTTP 指南](docs/SYSTEM_ONE_API.zh-CN.md)介绍服务配置、认证与兼容边界。

## 文档

[结果文档](docs/RESULTS.zh-CN.md)介绍速度、质量、扰动测试和限制。[方法文档](docs/METHOD.zh-CN.md)记录固定 prompt、指标和计时范围。[复现指南](docs/REPRODUCE.zh-CN.md)提供固定环境和验证命令。[基准包](benchmarks/README.zh-CN.md)包含 fixture、runner 和来源选择。[交互回放](demo/index.html)与[纯浏览器 WebGPU demo](webgpu-demo/index.html)用于可视化浏览项目。

## 使用边界与来源

FastJev 返回以所给选项为条件的概率，并标记 `calibrated=False`。实际部署工作负载上的验证与校准用于建立高影响自动化阈值。实验性的共享前缀模式可能改变接近决策边界的 BF16 argmax。

模型权重保存在上游站点，第三方评估记录保存在原始来源。上游模型沿用各自许可证，精确 revision 记录在[第三方清单](THIRD_PARTY.zh-CN.md)。

[第三方清单](THIRD_PARTY.zh-CN.md)记录代码与模型来源。

项目代码按 [MIT 许可证](LICENSE)发布。
