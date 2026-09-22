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

FastJev 是面向自部署的 [Jev](https://docs.typesafe.ai/) 开源实现，也是 [SemIf](https://github.com/TheoLeeCJ/SemIf) 的持续维护分支。固定 revision 的开放模型通过 Torch、vLLM、MLX、llama.cpp 和可选 EXL3 量化运行 `Choice`、`Boolean` 和 `Score` 接口。WebGPU demo 提供浏览器本地推理。

## 为什么选择 FastJev？

FastJev 把 Jev 自部署收敛为标准 Python 工作流。SDK 负责加载模型、验证 2 至 16 个选项、执行评分，并返回包含概率、token 用量、耗时、模型 revision 和 prompt 版本的类型化结果。

常驻 Torch、批量 vLLM、原生 MLX、llama.cpp GGUF、EXL3、CLI 和可选的 System One 兼容 HTTP API 共享同一结果模型。Torch、MLX、llama.cpp 和 EXL3 在一次评分中返回结果，输出 token 数量为 0。

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

默认后端需要 Python 3.10+，以及恰好一块可见 CUDA GPU 或 Apple Silicon MPS。`device="auto"` 会优先选择 CUDA，否则选择 MPS。Qwen3.5-4B 源 checkpoint 占用约 9 GB 磁盘。历史 [RTX 5090 SDK 冒烟测试](https://github.com/chengyongru/fastjev/pull/5)测得峰值 GPU 分配为 7.891 GiB。

```bash
python -m venv .venv
. .venv/bin/activate
export HF_HOME=/path/to/large-drive/huggingface
pip install 'fastjev[torch]'
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

使用本地或 Hugging Face 托管的 GGUF 文件时安装 `fastjev[llama-cpp]`。[Python SDK 指南](docs/SDK.zh-CN.md)介绍 Apple Silicon、EXL3、llama.cpp 前缀复用、校准与来源记录。

### 安装最新源码

需要运行 `main` 的最新代码时使用可编辑安装。

```bash
git clone https://github.com/chengyongru/fastjev.git
cd fastjev
pip install -e '.[torch]'
```

## 实测结果

### RTX 5090 后端性能对比

三项测量都在 WSL2 的同一块 RTX 5090 上，通过公共 `decide_many` API 运行
Qwen3.5-4B；每项测量包含一次 warmup 和七次三问题调用。

| 后端 | 模型格式 | 请求执行方式 | 模型加载 | 三项决策中位耗时 | 决策/秒 |
|---|---|---|---:|---:|---:|
| Torch | BF16 | 顺序执行 | 10.04 秒 | 161.97 毫秒 | 18.52 |
| vLLM | BF16 | 批量执行 | 40.43 秒 | **82.73 毫秒** | **36.26** |
| llama.cpp | Q4_K_M | 顺序执行 | **4.05 秒** | 2.344 秒 | 1.28 |

在这项 RTX 5090 / WSL 工作负载上，vLLM 吞吐量为 Torch 的 1.96 倍，batch 中位延迟降低 48.9%。额外的 30.38 秒启动成本需要常驻处理约 383 个三项决策 batch 才能抵消。两个后端的三项选择完全一致。

[PR #7](https://github.com/chengyongru/fastjev/pull/7) 记录了这项历史集成测量，包括精确环境和汇总中位数。仓库内的可复现 benchmark bundle 覆盖另外提交了逐行数据的实验。

llama.cpp 一行使用后续的 shell safeguard fixture，输入为 183、185 和 165 token；
Torch 与 vLLM 两行共享另一组完全相同的 125、152 和 151-token prompt。因此该表可以
展示实际 SDK 成本，但不能据此计算 llama.cpp 与 BF16 的精确速度倍率。llama.cpp 的
全部决策均符合预期，输出 token 数为 0。[结果文档](docs/RESULTS.zh-CN.md#rtx-5090-上的-llamacpp-q4_k_m)
记录了质量结果、精确条件和逐行证据。

可选的 27B EXL3 后端也在同一张 RTX 5090 上用相同的三问 safeguard 单独验证：
全部选择符合预期、输出 token 数为 0，调用中位耗时为 365.48 毫秒。由于模型大小、
量化方式、runtime 和后台 GPU 占用均不同，它不进入上面的同模型表。[结果文档](docs/RESULTS.zh-CN.md#rtx-5090-上的公共-sdk-shell-safeguard-验证)
还记录了校准 Torch smoke 与 llama.cpp 前缀复用对照。

### 决策质量

| 冻结工作负载 | FastJev 直接 Qwen3.5-4B | EXL3 Qwen3.8-27B | Qwen3-Reranker-4B | 已发布 Jev |
|---|---:|---:|---:|---:|
| 自编决策，144 行 | 0.813 | **0.946** | 0.625 | N/A |
| WANLI，256 行 | **0.637** | N/A | 0.522 | N/A |
| TypeSafe 公开子集，102 行 / 20 个 case | **0.845** | N/A | 0.560 | 0.883 |

前两行使用平衡准确率；自编数据指标会对三个决策族等权平均。第三行使用按 case
等权的众数一致率。EXL3 使用相同自编行和指标，但属于系统级比较：模型族、规模、
量化方式和 runtime 均不同。Jev 一列复述公开记录，开放模型数值来自本地冻结评估。
[结果文档](docs/RESULTS.zh-CN.md)收录完整数据、扰动测试与声明边界。

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
| PyTorch/CUDA | `pip install 'fastjev[torch]'` | 默认直接 logits 评分 |
| PyTorch/MPS | `pip install 'fastjev[torch]'` | Apple Silicon 原生 Transformers 推理 |
| vLLM/CUDA | `pip install 'fastjev[vllm]'` | 批量常驻服务 |
| MLX/Apple Silicon | 查看 [MLX 指南](docs/MLX.zh-CN.md) | 原生 macOS arm64 推理 |
| llama.cpp/GGUF | `pip install 'fastjev[llama-cpp]'` | 本地或 Hugging Face 托管的 GGUF 文件 |
| ExLlamaV3/EXL3 | `pip install 'fastjev[exl3]'` | 在 NVIDIA GPU 上运行更大的量化模型 |
| WebGPU/GGUF | 打开[浏览器 demo](webgpu-demo/index.html) | 浏览器本地推理 |

使用 `fastjev-score` 处理 JSONL。安装 `fastjev[api,torch]` 并运行 `fastjev-serve`，即可提供 `POST /v1/systemone` 和 `GET /v1/models`。[SDK 指南](docs/SDK.zh-CN.md)介绍 batching 和自定义后端。[HTTP 指南](docs/SYSTEM_ONE_API.zh-CN.md)介绍服务配置、认证与兼容边界。

集成时应导入 `fastjev` 并使用 `fastjev-*` 命令。只有文档列出的模块属于受支持的
公共 API。

## 文档

[结果文档](docs/RESULTS.zh-CN.md)介绍速度、质量、扰动测试和限制。[校准报告](docs/CALIBRATION.zh-CN.md)介绍按工作负载绑定的温度缩放。[方法文档](docs/METHOD.zh-CN.md)记录固定 prompt、指标和计时范围。[复现指南](docs/REPRODUCE.zh-CN.md)提供固定环境和验证命令。[基准包](benchmarks/README.zh-CN.md)包含 fixture、runner 和来源选择。[交互回放](demo/index.html)与[纯浏览器 WebGPU demo](webgpu-demo/index.html)用于可视化浏览项目。

## 使用边界与来源

FastJev 返回以所给选项为条件的概率。只有显式附加与身份绑定的 `TemperatureCalibration` 后，结果才会标记为已校准；不得跨工作负载、backend、模型 revision 或 prompt version 复用校准参数。Torch 的实验性共享前缀模式可能改变接近决策边界的 BF16 argmax；llama.cpp 前缀复用同样需要显式开启，默认关闭。

模型权重保存在上游站点，第三方评估记录保存在原始来源。上游模型沿用各自许可证，精确 revision 记录在[第三方清单](THIRD_PARTY.zh-CN.md)。

[第三方清单](THIRD_PARTY.zh-CN.md)记录代码与模型来源。

项目代码按 [MIT 许可证](LICENSE)发布。
