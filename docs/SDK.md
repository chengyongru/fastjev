# Python SDK

[English](SDK.md) | [简体中文](SDK.zh-CN.md)

`FastJev` is the stable decision interface. It owns one resident `ScoringBackend`, validates typed questions before inference, and converts backend results into typed decisions with probability, usage, timing, provenance, and uncertainty metadata.

## Load the built-in Torch backend

Install the project with the Torch dependencies when using CUDA or Apple MPS. A third-party backend can install the dependency-free core package instead.

```bash
pip install 'fastjev[torch]'
```

`from_pretrained` is a convenience constructor for the direct-logit Torch backend:

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

`device="auto"` prefers a single visible CUDA GPU and otherwise selects MPS. Select
Apple Silicon explicitly when deployment should fail instead of falling back:

```python
jev = FastJev.from_pretrained(
    "Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    device="mps",
    dtype="float16",
)
```

Supported dtypes are `bfloat16`, `float16`, and `float32`. Model and macOS support
still determine which combination is practical. Direct, serial, and shared option
scoring support MPS; reranker mode remains CUDA-only. Shared scoring uses independent
batch-one suffixes on MPS because that execution shape is faster on Apple GPUs.

Use `FastJev` as a context manager when its lifetime is scoped. Closing an engine closes its backend and rejects later decisions; built-in backends release their model and tokenizer references without modifying global accelerator state.

## Evaluate multiple states

`decide_many(state, questions)` evaluates several questions for one state.
`decide_batch(states, questions)` applies the same question mapping to a sequence
of states and returns one decision dictionary per state, in input order. Dictionary
keys and `Decision.id` retain the supplied question IDs. Strings and dictionaries
must be wrapped in a sequence; an empty sequence returns `[]` with a valid schema.
All state/schema validation precedes scoring. Prompt token limits are checked by
the backend during encoding; no partial result is returned if a later prompt fails.

```python
from fastjev import Boolean, FastJev

with FastJev.from_pretrained(
    "Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    batch_size=8,
    sort_by_length=True,
) as jev:
    results = jev.decide_batch(
        ["Please refund the duplicate charge.", "I cannot log in."],
        {"refund": Boolean("Does the customer request a refund?")},
    )
    print([item["refund"].value for item in results])
```

Torch's `batch_size` bounds **decision prompts per forward pass**, not states;
each state contributes one prompt per question. Its default is `1` (sequential).
The same options are available on `TorchBackend` and apply to `decide_many` too.
`sort_by_length=True` groups prompts within windows of up to eight batches and
restores result order. It reduces padding for mixed lengths but buffers more
encoded input. The SDK materializes the request/result list; split very large
collections into application-level chunks.

Torch uses left padding, attention masks, and unpadded token positions. Input usage
excludes padding; output tokens remain zero. Larger batches use more memory and can
change reduced-precision probabilities or close argmax decisions. Validate the
batch configuration on the deployment workload, including any calibration profile;
batching does not establish calibration. OOM errors propagate without automatic retry.
Other backends accept the same API and control their own execution strategy.

For Torch tensor batches, each decision's `forward_seconds` is the shared batch
forward duration; `total_seconds` adds the shared window encoding duration. These
values are not additive or individual request latencies. Measure the public call's
wall time for throughput. The benchmark in `benchmarks/cross_state_batching.py`
records sequential, batched, and length-grouped calls with per-decision evidence.

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

For a `decide_many` call whose questions share the same exact state, enable explicit
llama.cpp state-prefix reuse:

```python
backend = LlamaCppBackend.from_pretrained(
    "bartowski/Qwen_Qwen3.5-4B-GGUF",
    revision="4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    filename="Qwen_Qwen3.5-4B-Q4_K_M.gguf",
    n_gpu_layers=-1,
    prefix_reuse=True,
)
```

The backend evaluates the state prefix once, saves the llama.cpp sequence state, and
restores it before each question suffix. The default remains full-prompt direct
scoring, so an application opts into the memory and latency tradeoff deliberately.

## Load the optional EXL3 backend

EXL3 uses ExLlamaV3 to run larger quantized checkpoints on NVIDIA GPUs while keeping
the same typed SDK and zero-output-token option readout:

```bash
pip install 'fastjev[exl3]'
```

The PyPI ExLlamaV3 package builds its CUDA extension. On deployment hosts, the
[matching prebuilt release wheel](https://github.com/turboderp-org/exllamav3/releases)
is usually faster and more predictable to install; select the wheel for the host's
Python, PyTorch, CUDA, OS, and architecture before installing FastJev.

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

`gpu_split` is the single visible GPU allocation in GiB. Remote repositories require
an immutable 40-character revision; local EXL3 directories require a provenance
label. FastJev renders the model's Hugging Face chat template, validates exact
single-token answer slots and boundary stability, and refuses prompts beyond either
the public input limit or the EXL3 cache capacity.

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

## Calibrate one deployment workload

Raw option scores are conditional on the declared options, not universal confidence.
`TemperatureCalibration` applies a single positive temperature only after verifying
the backend, model, revision, and prompt version used to fit it:

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

Fit on labeled examples representative of the production workload and evaluate on
held-out groups. Temperature scaling changes confidence but preserves option ordering.
The returned `Decision.calibration` records the method, temperature, and workload;
`uncertainty.calibrated` becomes `True`. Identity or prompt mismatches fail instead of
silently applying an unrelated profile. The [calibration report](CALIBRATION.md)
contains the offline cross-validation method and frozen evidence.

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

`FastJev` serializes calls into a backend so one resident model is not used concurrently. The vLLM backend batches all requests received by one `decide_many` call. EXL3 evaluates them in order. Torch and MLX expose separate experimental shared-prefix modes; llama.cpp reuses an exact state prefix when `prefix_reuse=True`.

## Backend protocol

The engine does not import Torch, MLX, Transformers, llama.cpp, ExLlamaV3, or vLLM. It depends on the runtime-checkable `ScoringBackend` protocol:

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

Built-in implementations are available as `TorchBackend`, `MLXBackend`, `LlamaCppBackend`, `ExLlamaV3Backend`, and `VLLMBackend`. `FastJev.from_pretrained` remains a Torch convenience; other runtimes use explicit dependency injection.

## Package namespace

Use `fastjev` and the documented public modules for integrations. Backend
adapters live under `fastjev.backends`; low-level inference code lives under
the private `fastjev._runtime` package and is not a supported integration
surface. Use `fastjev-score` and `fastjev-serve` for command-line and HTTP
entry points.

## Result semantics

Every `Decision` exposes:

- `value` and `selected`;
- the complete `probabilities` mapping;
- input/output token `usage`;
- backend `timing` when supplied;
- model, revision, backend, prompt version, and probability-status `provenance`;
- normalized distribution entropy in `uncertainty`.
- applied temperature metadata in `calibration`, or `None` for raw scores.

The distribution is conditional on the declared options. `uncertainty.normalized_entropy` ranges from zero for a peaked distribution to one for a uniform distribution. `uncertainty.calibrated` is `False` unless an identity-bound profile was applied. Validate thresholds on the deployment workload before automating consequential actions.

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
