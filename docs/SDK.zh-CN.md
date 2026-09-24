# Python SDK

[English](SDK.md) | 简体中文

`FastJev` 是稳定的决策接口。它持有一个常驻 `ScoringBackend`，在推理前验证类型化问题，并将 backend 结果转换为包含概率、用量、计时、来源和不确定性元数据的类型化决策。

## 加载内置 Torch backend

使用 CUDA 或 Apple MPS 路径时，安装项目及 Torch 依赖。第三方 backend 可以只安装无依赖的核心包。

```bash
pip install 'fastjev[torch]'
```

`from_pretrained` 是直接 logits Torch backend 的便利构造器：

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

`device="auto"` 会优先选择一块可见 CUDA GPU，否则选择 MPS。如果部署必须使用
Apple Silicon，显式指定设备可以避免意外回退：

```python
jev = FastJev.from_pretrained(
    "Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    device="mps",
    dtype="float16",
)
```

支持 `bfloat16`、`float16` 和 `float32`；实际可用组合仍取决于模型与 macOS。
direct、serial 和 shared 选项评分支持 MPS，reranker 仍只支持 CUDA。shared 模式在
MPS 上使用独立的 batch-one suffix，因为该执行形态在 Apple GPU 上更快。

当 engine 生命周期有明确作用域时，可将 `FastJev` 用作 context manager。关闭 engine 会关闭 backend，并拒绝后续决策；内置 backend 会释放模型与 tokenizer 引用，但不修改全局 accelerator 状态。

## 批量处理多个 state

`decide_many(state, questions)` 对一个 state 回答多个问题；
`decide_batch(states, questions)` 对一组 state 应用同一问题映射，按输入顺序返回
决策字典列表。字典键和 `Decision.id` 保留原始问题 ID。字符串和对象需要放入序列，
空序列在 schema 有效时返回 `[]`。所有 state/schema 都在评分前验证；token 上限由
backend 在编码时检查，后续 prompt 失败时不会返回部分结果。

```python
from fastjev import Boolean, FastJev

with FastJev.from_pretrained(
    "Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    batch_size=8,
    sort_by_length=True,
) as jev:
    results = jev.decide_batch(
        ["请退还重复扣款。", "我无法登录。"],
        {"refund": Boolean("客户是否要求退款？")},
    )
    print([item["refund"].value for item in results])
```

Torch 的 `batch_size` 限制每次前向的**决策 prompt 数量**，不是 state 数量；每个
state 的每个问题各占一个 prompt。默认值为 `1`，保持顺序评分。`TorchBackend`
也接受这些配置，配置同样作用于 `decide_many`。`sort_by_length=True` 在最多八个
batch 的窗口内按 prompt 长度分组，之后恢复原始顺序；这样能减少混合长度输入的
padding，但会缓存更多编码输入。SDK 会构造完整的请求和结果列表，超大数据集应由
应用分段调用。

Torch 使用左侧 padding、attention mask 和不含 padding 的 token 位置。输入用量
不计 padding，输出 token 数仍为零。增大 batch 会增加显存占用，也可能改变低精度
概率及接近边界的 argmax。应在部署工作负载上验证 batch 配置及校准 profile；批处理
本身不代表已校准。OOM 会直接抛出，不自动重试。其他 backend 使用相同 API，但各自
决定执行方式。

Torch 张量批处理中的 `forward_seconds` 是共享的 batch 前向耗时；`total_seconds`
加上共享的窗口编码耗时。这些值不能逐行相加，也不是单请求延迟。吞吐应使用公共调用
的整体墙钟时间。`benchmarks/cross_state_batching.py` 会记录顺序、普通批处理和长度
分组三种模式，并保存逐决策结果。

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

如果一次 `decide_many` 的所有问题共享完全相同的 state，可显式开启 llama.cpp
state-prefix 复用：

```python
backend = LlamaCppBackend.from_pretrained(
    "bartowski/Qwen_Qwen3.5-4B-GGUF",
    revision="4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    filename="Qwen_Qwen3.5-4B-Q4_K_M.gguf",
    n_gpu_layers=-1,
    prefix_reuse=True,
)
```

backend 只计算一次 state 前缀，保存 llama.cpp sequence state，并在每个问题 suffix
前恢复它。默认仍为完整 prompt direct 评分，应用需要明确接受这项内存与延迟取舍。

## 加载可选 EXL3 backend

EXL3 通过 ExLlamaV3 在 NVIDIA GPU 上运行更大的量化 checkpoint，同时保持相同的
类型化 SDK 和零输出 token 选项读数：

```bash
pip install 'fastjev[exl3]'
```

PyPI 上的 ExLlamaV3 会构建 CUDA extension。部署机器通常更适合安装
[匹配的预编译 release wheel](https://github.com/turboderp-org/exllamav3/releases)：
先按 Python、PyTorch、CUDA、操作系统与架构选择 wheel，再安装 FastJev。

```python
from fastjev import ExLlamaV3Backend, FastJev

backend = ExLlamaV3Backend.from_pretrained(
    "turboderp/Qwen3.8-27B-exl3",
    revision="a35e75a73baee51da709329d19294245cbeeb5d8",
    cache_size=16384,
    gpu_split=22.5,
)

with FastJev(backend) as jev:
    answers = jev.decide_many(state, questions)
```

`gpu_split` 是分配给单块可见 GPU 的 GiB 数。远程仓库必须提供不可变的 40 字符
revision，本地 EXL3 目录则提供 provenance 标签。FastJev 使用模型的 Hugging Face
chat template，验证答案槽位是边界稳定的精确单 token，并拒绝超过公共输入上限或
EXL3 cache 容量的 prompt。

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

## 校准一个部署工作负载

原始选项得分只以声明的选项为条件，并不是通用置信度。`TemperatureCalibration`
只会在核对用于拟合的 backend、模型、revision 与 prompt version 后应用一个正温度：

```python
from fastjev import CalibrationSample, FastJev, TemperatureCalibration

profile = TemperatureCalibration.fit(
    [
        CalibrationSample((0.90, 0.10), correct_index=0),
        CalibrationSample((0.80, 0.20), correct_index=1),
    ],
    workload="support-routing-v3",
    backend=backend.info.name,
    model=backend.info.model,
    revision=backend.info.revision,
    prompt_version="direct-options-v1",
)

jev = FastJev(backend, calibration=profile)
```

应使用能代表生产工作负载的有标签样本拟合，并在分组隔离的留出集上评估。温度缩放
改变置信度，但不改变选项顺序。返回的 `Decision.calibration` 会记录方法、温度与
workload，`uncertainty.calibrated` 变为 `True`。身份或 prompt 不匹配时会直接失败，
不会静默套用无关 profile。[校准报告](CALIBRATION.zh-CN.md)收录离线交叉验证方法与
冻结证据。

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

`FastJev` 会串行调用 backend，避免并发使用同一个常驻模型。vLLM backend 会批量处理一次 `decide_many` 收到的所有请求；EXL3 会按顺序执行。Torch 与 MLX 提供独立的实验性 shared-prefix 模式；llama.cpp 在 `prefix_reuse=True` 时复用完全相同的 state 前缀。

## Backend 协议

engine 不导入 Torch、MLX、Transformers、llama.cpp、ExLlamaV3 或 vLLM，只依赖可在运行时检查的 `ScoringBackend` 协议：

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

内置实现包括 `TorchBackend`、`MLXBackend`、`LlamaCppBackend`、`ExLlamaV3Backend` 和 `VLLMBackend`。`FastJev.from_pretrained` 继续作为 Torch 便利入口；其他 runtime 通过显式依赖注入接入。

## 包命名空间

集成时使用 `fastjev` 及文档列出的公共模块。backend 适配器位于
`fastjev.backends`；底层推理代码位于私有的 `fastjev._runtime` 包中，不属于受支持的
集成接口。命令行和 HTTP 入口分别使用 `fastjev-score` 与 `fastjev-serve`。

## 结果语义

每个 `Decision` 都提供：

- `value` 和 `selected`；
- 完整 `probabilities` mapping；
- 输入/输出 token `usage`；
- backend 提供时的 `timing`；
- 模型、revision、backend、prompt version 和概率状态 `provenance`；
- `uncertainty` 中的归一化分布熵。
- 已应用的温度元数据 `calibration`；原始得分为 `None`。

分布以声明的选项为条件。`uncertainty.normalized_entropy` 从尖锐分布的零变化到均匀分布的一。只有应用了身份绑定的 profile 时，`uncertainty.calibrated` 才为 `True`。在自动执行重要操作前，请在部署工作负载上验证阈值。

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
