# Python SDK

[English](SDK.md) | 简体中文

`FastJev` 是稳定的决策接口。它持有一个常驻 `ScoringBackend`，在推理前验证类型化问题，并将 backend 结果转换为包含概率、用量、计时、来源和不确定性元数据的类型化决策。

## 加载内置 Torch backend

使用内置 CUDA 路径时，安装项目及 Torch 依赖。第三方 backend 可以只安装无依赖的核心包。

```bash
pip install 'fastjev[torch]'
```

`from_pretrained` 是直接 logits Torch/CUDA backend 的便利构造器：

```python
from fastjev import Choice, FastJev, Option

jev = FastJev.from_pretrained(
    "Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
)

result = jev.decide(
    state={"message": "The customer cannot access the account."},
    question=Choice(
        "Which queue should handle this request?",
        [
            Option("access", "Account access and authentication support."),
            Option("billing", "Billing, payments, and refunds."),
        ],
    ),
)

print(result.value)
print(result.probabilities)
jev.close()
```

当 engine 生命周期有明确作用域时，可将 `FastJev` 用作 context manager。关闭 engine 会关闭 backend，并拒绝后续决策；内置 backend 会释放模型与 tokenizer 引用，但不修改全局 accelerator 状态。

## 通过 llama.cpp 加载 GGUF

使用本地 GGUF（包括桌面模型管理器下载的文件）或 Hugging Face 上托管的 GGUF 时，安装可选的 llama.cpp Python binding：

```bash
pip install 'fastjev[llama-cpp]'
```

使用 NVIDIA GPU offload 时，请从与 CUDA runtime 匹配的上游 wheel index 安装。
已提交的 RTX 5090 验证使用 CUDA 13.0：

```bash
pip install 'fastjev[llama-cpp]' \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130
```

`LlamaCppBackend` 会直接读取 GGUF 内置的 chat template 和最后位置 logits，不启动也不依赖其他服务。缺少安全 chat template 或无法将答案槽位编码为精确单 token 的 GGUF 会被拒绝：

```python
from fastjev import Choice, FastJev, LlamaCppBackend, Option

backend = LlamaCppBackend.from_pretrained(
    r"C:\models\qwen3.5-4b-instruct-q4_k_m.gguf",
    revision="local-qwen3.5-4b-q4-k-m",
    n_gpu_layers=-1,
)

with FastJev(backend) as jev:
    result = jev.decide(
        "客户无法访问账户。",
        Choice("哪个队列应该处理这个请求？", [
            Option("access", "账户访问和身份验证支持。"),
            Option("billing", "账单、支付和退款支持。"),
        ]),
    )
```

要从 Hugging Face 自动下载，请传入仓库 ID、精确的 GGUF 文件名，以及不可变的 40 位 commit revision。下载使用 Hugging Face 的标准缓存和认证设置：

```python
backend = LlamaCppBackend.from_pretrained(
    "bartowski/Qwen_Qwen3.5-4B-GGUF",
    revision="4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    filename="Qwen_Qwen3.5-4B-Q4_K_M.gguf",
    n_gpu_layers=-1,
)
```

显式文件名可以避免 fastjev 在仓库的多个量化版本之间擅自选择。本地文件的 `revision` 是 provenance 标签；两种来源都会记录解析后的工件路径和 GGUF SHA-256。量化 GGUF 的质量和延迟需要独立于 BF16 Torch baseline 重新验证。

## 加载可选 vLLM backend

在受支持的 CUDA 主机上安装独立固定版本的 vLLM runtime：

```bash
pip install 'fastjev[vllm]'
```

显式构造 backend，再注入同一个 `FastJev` 接口：

```python
from fastjev import Choice, FastJev, Option, VLLMBackend

backend = VLLMBackend.from_pretrained(
    "Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    gpu_memory_utilization=0.8,
)

with FastJev(backend) as jev:
    result = jev.decide(
        {"message": "The customer cannot access the account."},
        Choice("Which queue should handle this request?", [
            Option("access", "Account access and authentication support."),
            Option("billing", "Billing, payments, and refunds."),
        ]),
    )
```

远程模型 ID 必须使用固定的 40 字符 Hugging Face revision。本地模型路径必须提供非空 revision 标签，以便记录结果来源。除 fastjev 管理的模型标识、revision、信任策略和 context 上限外，其他关键字参数会传给 `vllm.LLM`。

一次 `decide_many` 会转换为一次批量 `LLM.generate` 调用。backend 渲染并验证与 Torch 路径相同的直接决策 prompt，只允许生成单 token 的答案槽位，并请求所有已声明槽位的 log probability。返回值是这些槽位的条件 next-token score，不需要解析生成文本。vLLM 会为每个问题生成一个受约束 token，因此每项结果记录一个 output token。

### WSL 兼容性

vLLM 0.29.0 的 V2 model runner 依赖 CUDA Unified Virtual Addressing（UVA），而 WSL 可能无法提供它。如果启动时报错 `UVA is not available`，请在 Python 导入 vLLM 前切换到 V1 runner：

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

如果 FlashInfer sampling JIT 报告 CUDA compiler 与 toolkit headers 不兼容，请使用 vLLM 原生 sampler：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

这些开关只选择 vLLM 内部实现，不会改变 fastjev backend 合约或决策语义。

## 类型化问题

所有问题都会编译为同一种 backend-neutral categorical request：

- `Choice` 返回获胜选项的稳定 ID。
- `Boolean` 返回 `True` 或 `False`。
- `Score` 返回按概率加权的数值，并通过 `selected` 暴露获胜 level。

`decide_many` 接受以稳定问题 ID 为键的 mapping，并通过一次 backend 调用提交所有问题：

```python
from fastjev import Boolean, Level, Score

answers = jev.decide_many(state, {
    "urgent": Boolean("Does this require immediate action?"),
    "severity": Score("How severe is it?", [
        Level(0, "Minor"),
        Level(1, "Degraded"),
        Level(2, "Blocking"),
    ]),
})
```

`FastJev` 会串行调用 backend，避免并发使用同一个常驻模型。vLLM backend 会批量处理一次 `decide_many` 收到的所有请求；内置 Torch、MLX 和 llama.cpp backend 当前仍按顺序评估这些请求。

## Backend 协议

engine 不导入 Torch、MLX、Transformers、llama.cpp 或 vLLM，只依赖可在运行时检查的 `ScoringBackend` 协议：

```python
from typing import Sequence

from fastjev.backends import (
    BackendCapabilities,
    BackendInfo,
    BackendRequest,
    BackendResult,
)


class RemoteBackend:
    @property
    def info(self) -> BackendInfo: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def score(self, requests: Sequence[BackendRequest]) -> Sequence[BackendResult]: ...

    def close(self) -> None: ...
```

直接注入实现：

```python
from fastjev import FastJev

jev = FastJev(RemoteBackend(...))
```

backend 负责模型加载、prompt 执行、batching 和资源清理。它必须按请求顺序为每个请求返回一个 `BackendResult`，并保持精确的请求 ID 与选项 ID。概率必须有限、非负且总质量大于零；`FastJev` 会对其归一化，并以 `BackendProtocolError` 拒绝格式错误的结果。

backend 特定的选项限制写入 `BackendCapabilities`。领域类型本身不嵌入 Torch 内置上限，因此未来 backend 可以支持不同选项数量，而无需改变公共决策 API。

内置实现包括 `TorchBackend`、`MLXBackend`、`LlamaCppBackend` 和 `VLLMBackend`。`FastJev.from_pretrained` 继续作为 Torch 便利入口；其他 runtime 通过显式依赖注入接入。

## 结果语义

每个 `Decision` 都提供：

- `value` 和 `selected`；
- 完整 `probabilities` mapping；
- 输入/输出 token `usage`；
- backend 提供时的 `timing`；
- 模型、revision、backend、prompt version 和概率状态 `provenance`；
- `uncertainty` 中的归一化分布熵。

分布以声明的选项为条件。`uncertainty.normalized_entropy` 从尖锐分布的零变化到均匀分布的一，且 `uncertainty.calibrated` 为 `False`。在自动执行重要操作前，请在部署工作负载上验证阈值。

## System One 兼容层

System One 是适配器，不是 SDK 的领域模型：

```python
from fastjev import SystemOneAdapter
from fastjev.http import create_app

service = SystemOneAdapter(
    jev,
    served_model="fastjev-qwen3.5-4b",
    description="fastjev direct option-logit baseline",
    release_date="2026-09-18",
)
app = create_app(service)
```

适配器将 `noul`、`choice` 和 `score` 请求映射到同一个 `FastJev` engine。HTTP 边界、TypeSafe 命名和 vendor-compatible wire shape 都不会进入 backend 协议。
