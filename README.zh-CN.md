<p align="center">
  <img src="./assets/fastjev-logo.svg"
       width="420"
       alt="FastJev" />
</p>

<div align="center">

[English](README.md) | 简体中文

<img src="assets/fastjev-cover.webp" alt="FastJev 自托管语义决策 SDK" width="100%">

**把非结构化状态直接变成类型化决策，无需生成回答。**

</div>

FastJev 是处理 AI 系统中“小决策”的开源 Python SDK：*应该分到哪个队列？是否允许执行这个动作？现有证据是否充分？风险有多高？* 它使用自托管开放模型评估运行时定义的 `Choice`、`Boolean` 和 `Score` 问题，返回稳定值与完整选项分布。

这是语义决策推理，不是聊天框架，也不是任意数据抽取工具。每次请求都可以使用新的标准和选项，无需为具体任务准备训练集、固定标签分类头或路由样例库。

## 可以用它做什么？

| 应用 | 示例决策 | 类型化结果 |
|---|---|---|
| Agent 安全门 | “这个 shell 命令会破坏持久数据吗？” | `Boolean` |
| 客服或邮件分流 | “哪个团队应该处理这个请求？” | `Choice` |
| 证据检查 | “记录支持、反驳还是没有提及这项主张？” | `Choice` |
| 风险与优先级 | “这项事件有多严重？” | `Score` |
| 模型或工具路由 | “下一步需要哪项能力？” | `Choice` |

每项结果都包含选中值、全部声明选项的概率、token 用量、耗时、模型 revision、prompt 版本，以及明确的“概率未校准”标记。

## 为什么选择 FastJev？

评分原理并不神秘：因果语言模型可以暴露 next-token logits。FastJev 提供的是一次性 logprob 调用不具备的应用层：

- **无需训练：** 运行时直接提交新的标准和选项说明，不必收集路由样例或微调分类器。
- **不走 JSON 回答路径：** Torch 和 MLX 直接读取声明选项的 logits，输出 token 为零，不需要 schema 修复、重试或解析循环。
- **稳定的类型协议：** 输入验证、2–16 个选项槽位、归一化分布、`Choice`/`Boolean`/`Score` 结果和一致的错误边界。
- **完整的部署边界：** 常驻 Torch、批量 vLLM、原生 MLX、CLI 和可选的 System One 兼容 HTTP API 使用同一结果模型。
- **运行可审计：** 固定模型 revision，并保留 prompt hash、逐行预测、原始计时和校验和。

如果只需要一个二元 prompt 和原始 logprobs，直接调用
[llama.cpp](https://github.com/ggml-org/llama.cpp) 更简单。当这些判断成为必须保持类型稳定、可迁移、可测试、可追溯的应用接口时，FastJev 才能体现价值。

## 与相邻项目的区别

这些项目解决的是相邻问题。下表比较能力边界，不是跨项目速度 benchmark。

| 项目 | 核心机制 | 适合选择它的场景 | FastJev 的区别 |
|---|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | 小型、专用的非自回归决策模型 | 低资源或多语言部署，尤其是愿意按工作负载微调时 | 使用标准开放因果模型，无需训练即可处理运行时决策，默认输入上限为 4,096 token |
| [semantic-router](https://github.com/aurelio-labs/semantic-router) | 路由样例的 embedding 相似度 | 路由与样例稳定，向量相似度足够解决问题时 | 每次请求都根据完整状态、新标准和选项含义作出判断 |
| [Outlines](https://github.com/dottxt-ai/outlines) | 受约束的自回归生成 | 需要任意 JSON、正则、grammar 或抽取 schema 时 | 专注较窄的决策场景，无需生成完整对象即可读取选项分数 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 底层本地推理与 token logprobs | 需要完全控制 runtime 或只做一次性分类器时 | 增加类型化问题、prompt/槽位验证、来源记录、多后端、HTTP 兼容和冻结评估 |

FastJev 并不适合所有工作负载：任意信息抽取应选择结构化生成工具；固定路由体系可选择 embedding router；如果已有训练数据且小型决策模型满足领域与延迟要求，就应使用相应的专用模型。

## 可以直接运行的模型

以下固定 checkpoint 都通过了相同的原生 BF16 直接 logits 接口验证。把任意模型 ID 与 revision 填入后面的快速开始代码即可：

| 模型 | 固定源 revision | 推荐场景 | 自编数据平衡准确率 | 浏览器选项 |
|---|---|---|---:|---:|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | 最小入门模型 | 0.440 | Q8_0，639 MB |
| [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | `12a3808a956f869c767195e9266b59c4d21d92e2` | 体积与质量平衡 | 0.686 | Q4_K_M，1.56 GB |
| **[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)** | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | **推荐；实测质量最佳** | **0.813** | Q4_K_M，3.01 GB |

质量数值来自原生 BF16 checkpoint，不代表浏览器量化模型。精确预测行、revision 和单独的浏览器冒烟结果见[模型梯度原始报告](results/raw/browser-model-ladder.json)。

## 快速开始

默认后端需要 Python 3.10+、CUDA，以及仅一块可见 GPU。Qwen3.5-4B 源 checkpoint 占用约 9 GB 磁盘；历史 [RTX 5090 SDK 冒烟测试](https://github.com/chengyongru/fastjev/pull/5)测得峰值 GPU 分配为 7.891 GiB。

```bash
git clone https://github.com/chengyongru/fastjev.git
cd fastjev
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

远程模型必须使用不可变的 40 字符 Hugging Face revision。本地模型目录也受支持，但需要提供非空 revision 标签来记录结果来源。

使用本地或 Hugging Face 托管的 GGUF 文件时安装 `.[llama-cpp]`。[Python SDK 指南](docs/SDK.zh-CN.md)介绍 llama.cpp 的设置与来源记录。

## 实测结果

### RTX 5090：Torch 与 vLLM

同机集成测试使用固定的 Qwen3.5-4B checkpoint、完全相同的三个问题输入（125、152 和 151 token）、一次 warmup 和七次 `decide_many` 计时：

| 后端 | 模型加载 | 三项决策 batch 中位耗时 | 决策/秒 |
|---|---:|---:|---:|
| Torch | **10.04 秒** | 161.97 毫秒 | 18.52 |
| vLLM | 40.43 秒 | **82.73 毫秒** | **36.26** |

在这项 RTX 5090 / WSL 工作负载上，vLLM 吞吐量为 Torch 的 1.96 倍，batch 中位延迟降低 48.9%。额外的 30.38 秒启动成本需要常驻处理约 383 个三项决策 batch 才能抵消。两个后端的三项选择完全一致。

这是一项记录在 [PR #7](https://github.com/chengyongru/fastjev/pull/7) 中的历史集成测量，不属于仓库内可复现的 benchmark bundle：PR 保存了精确环境和中位数，但没有提交七次计时的逐次原始值。

### 决策质量

| 冻结工作负载 | FastJev 直接 Qwen3.5-4B | Qwen3-Reranker-4B | 已发布 Jev |
|---|---:|---:|---:|
| 自编决策，144 行 | **0.813** | 0.625 | — |
| WANLI，256 行 | **0.637** | 0.522 | — |
| TypeSafe 公开子集，102 行 / 20 个 case | **0.845** | 0.560 | 0.883 |

前两行是平衡准确率，第三行是按 case 等权的众数一致率。Jev 数值来自公开记录，并非在线端点实测。完整数据、扰动测试与声明边界见[结果文档](docs/RESULTS.zh-CN.md)。

## 后端与接口

| Runtime | 安装 | 适用场景 |
|---|---|---|
| PyTorch/CUDA | `pip install -e '.[torch]'` | 默认直接 logits 评分 |
| vLLM/CUDA | `pip install -e '.[vllm]'` | 批量常驻服务 |
| MLX/Apple Silicon | 查看 [MLX 指南](docs/MLX.zh-CN.md) | 原生 macOS arm64 推理 |
| WebGPU/GGUF | 打开[浏览器 demo](webgpu-demo/index.html) | 无需 Python 的本地推理 |

使用 `fastjev-score` 处理 JSONL。安装 `.[api]` 并运行 `fastjev-serve`，即可提供 `POST /v1/systemone` 和 `GET /v1/models`。[SDK 指南](docs/SDK.zh-CN.md)介绍 batching 和自定义后端；[HTTP 指南](docs/SYSTEM_ONE_API.zh-CN.md)介绍服务配置、认证与兼容边界。

## 文档

- [结果](docs/RESULTS.zh-CN.md) — 速度、质量、扰动测试和限制
- [方法](docs/METHOD.zh-CN.md) — 固定 prompt、指标和计时范围
- [复现](docs/REPRODUCE.zh-CN.md) — 固定环境和验证命令
- [基准包](benchmarks/README.zh-CN.md) — fixture、runner 和来源选择
- [交互回放](demo/index.html)与[纯浏览器 WebGPU demo](webgpu-demo/index.html)

## 限制与来源

FastJev 返回的是以所给选项为条件的概率，并非经过校准的置信度。在使用阈值执行高影响自动化之前，请在实际部署工作负载上完成验证与校准。实验性的共享前缀模式可能改变接近决策边界的 BF16 argmax。

本仓库不分发模型权重或第三方评估记录。上游模型沿用各自许可证，精确 revision 记录在 [THIRD_PARTY.zh-CN.md](THIRD_PARTY.zh-CN.md)。

FastJev 是 [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf)（原名 OpenJev）的独立维护分支，保留原始 Git 历史和 MIT 许可证，但采用独立路线图。FastJev 与 TheoLeeCJ、TypeSafe 或 Jev 无隶属关系，也未获得其背书；它不复现 Jev 未公开的模型、训练方法、校准能力或性能。

项目代码按 [MIT 许可证](LICENSE)发布。
