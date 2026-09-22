# Python SDK

[English](SDK.md) | [简体中文](SDK.zh-CN.md)

`FastJev` is the stable decision interface. It owns one resident `ScoringBackend`, validates typed questions before inference, and converts backend results into typed decisions with probability, usage, timing, provenance, and uncertainty metadata.

## Load the built-in Torch backend

Install the project with the Torch dependencies when using the built-in CUDA path. A third-party backend can install the dependency-free core package instead.

```bash
pip install 'fastjev[torch]'
```

`from_pretrained` is a convenience constructor for the direct-logit Torch/CUDA backend:

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

Use `FastJev` as a context manager when its lifetime is scoped. Closing an engine closes its backend and rejects later decisions; built-in backends release their model and tokenizer references without modifying global accelerator state.

## Load a GGUF through llama.cpp

Install the optional llama.cpp Python bindings for a local GGUF file, including one downloaded through a desktop model manager, or for a GGUF hosted on Hugging Face:

```bash
pip install 'fastjev[llama-cpp]'
```

For NVIDIA GPU offload, install from the upstream wheel index matching the CUDA
runtime. The committed RTX 5090 validation used CUDA 13.0:

```bash
pip install 'fastjev[llama-cpp]' \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130
```

`LlamaCppBackend` reads the model's embedded chat template and last-position logits directly. It does not start or require another server, and it rejects GGUF files without a safe chat template or exact single-token answer slots:

```python
from fastjev import Choice, FastJev, LlamaCppBackend, Option

backend = LlamaCppBackend.from_pretrained(
    r"C:\models\qwen3.5-4b-instruct-q4_k_m.gguf",
    revision="local-qwen3.5-4b-q4-k-m",
    n_gpu_layers=-1,
)

with FastJev(backend) as jev:
    result = jev.decide(
        "The customer cannot access the account.",
        Choice("Which queue should handle this request?", [
            Option("access", "Account access and authentication support."),
            Option("billing", "Billing, payments, and refunds."),
        ]),
    )
```

For automatic Hugging Face download, pass the repository ID, exact GGUF filename, and immutable 40-character commit revision. The file uses the standard Hugging Face cache and authentication settings:

```python
backend = LlamaCppBackend.from_pretrained(
    "bartowski/Qwen_Qwen3.5-4B-GGUF",
    revision="4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    filename="Qwen_Qwen3.5-4B-Q4_K_M.gguf",
    n_gpu_layers=-1,
)
```

The explicit filename prevents fastjev from silently choosing among a repository's quantizations. For local files, `revision` is a provenance label. For both sources, `LlamaCppBackend` records the resolved artifact path and GGUF SHA-256. Quantized GGUF results require separate quality and latency validation from the BF16 Torch baseline.

## Load the optional vLLM backend

Install the separately pinned vLLM runtime on a supported CUDA host:

```bash
pip install 'fastjev[vllm]'
```

Construct the backend explicitly and inject it into the same `FastJev` interface:

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

Remote model IDs require a pinned 40-character Hugging Face revision. Local model paths require a nonempty revision label for result provenance. Additional keyword arguments are forwarded to `vllm.LLM`, except for the model identity, revision, trust policy, and context limit managed by fastjev.

One `decide_many` call becomes one batched `LLM.generate` call. The backend renders and validates the same direct decision prompts as the Torch path, permits only the single-token answer slots, and requests log probabilities for every declared slot. It then returns their conditional next-token scores without parsing generated text. vLLM generates one constrained token per question, so each result reports one output token.

### WSL compatibility

vLLM 0.29.0's V2 model runner requires CUDA Unified Virtual Addressing (UVA), which may be unavailable through WSL. If startup reports `UVA is not available`, select the V1 runner before Python imports vLLM:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

If FlashInfer sampling JIT reports incompatible CUDA compiler and toolkit headers, use vLLM's native sampler:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

These switches select vLLM implementations; they do not change the fastjev backend contract or decision semantics.

## Typed questions

All questions compile to one backend-neutral categorical request:

- `Choice` returns the winning stable option ID.
- `Boolean` returns `True` or `False`.
- `Score` returns the probability-weighted numeric value and exposes the winning level as `selected`.

`decide_many` accepts a mapping of stable question IDs and submits all questions in one backend call:

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

`FastJev` serializes calls into a backend so one resident model is not used concurrently. The vLLM backend batches all requests received by one `decide_many` call. The built-in Torch, MLX, and llama.cpp backends currently evaluate these requests in order.

## Backend protocol

The engine does not import Torch, MLX, Transformers, llama.cpp, or vLLM. It depends on the runtime-checkable `ScoringBackend` protocol:

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

Inject an implementation directly:

```python
from fastjev import FastJev

jev = FastJev(RemoteBackend(...))
```

A backend owns model loading, prompt execution, batching, and resource cleanup. It must return one `BackendResult` per request, in request order, with the exact request ID and option IDs. Probabilities must be finite, nonnegative, and have positive total mass; `FastJev` normalizes them and rejects malformed results with `BackendProtocolError`.

Backend-specific option limits belong in `BackendCapabilities`. The domain types themselves do not embed the built-in Torch limit, so a future backend may support a different number of options without changing the public decision API.

Built-in implementations are available as `TorchBackend`, `MLXBackend`, `LlamaCppBackend`, and `VLLMBackend`. `FastJev.from_pretrained` remains a Torch convenience; other runtimes use explicit dependency injection.

## Result semantics

Every `Decision` exposes:

- `value` and `selected`;
- the complete `probabilities` mapping;
- input/output token `usage`;
- backend `timing` when supplied;
- model, revision, backend, prompt version, and probability-status `provenance`;
- normalized distribution entropy in `uncertainty`.

The distribution is conditional on the declared options. `uncertainty.normalized_entropy` ranges from zero for a peaked distribution to one for a uniform distribution, and `uncertainty.calibrated` is `False`. Validate thresholds on the deployment workload before automating consequential actions.

## System One compatibility

System One remains an adapter rather than the SDK's domain model:

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

The adapter maps `noul`, `choice`, and `score` requests onto the same `FastJev` engine. The HTTP boundary, TypeSafe names, and vendor-compatible wire shape do not enter the backend protocol.
