<p align="center">
  <img src="./assets/fastjev-logo.svg"
       width="420"
       alt="FastJev" />
</p>

<div align="center">

[English](README.md) | [简体中文](README.zh-CN.md)

<img src="assets/fastjev-cover.webp" alt="FastJev self-hosted semantic decision SDK" width="100%">

**Turn unstructured state into typed decisions—without generating an answer.**

</div>

FastJev is an open-source Python SDK for the small decisions inside AI systems:
*Which queue? Should this action run? Is the evidence sufficient? How severe is
the risk?* It evaluates runtime-defined `Choice`, `Boolean`, and `Score`
questions with self-hosted open models and returns stable values plus the option
distribution.

This is semantic decision inference, not chat or arbitrary data extraction. The
criteria and options can change on every request, so no task-specific training
set, fixed label head, or route-utterance library is required.

## What can you build?

| Application | Example decision | Typed result |
|---|---|---|
| Agent safeguard | “Can this proposed shell command destroy durable state?” | `Boolean` |
| Support or email triage | “Which team should handle this request?” | `Choice` |
| Evidence gate | “Does this record support, refute, or omit the claim?” | `Choice` |
| Risk and priority | “How severe is this incident?” | `Score` |
| Model/tool routing | “Which capability is needed next?” | `Choice` |

Each result includes the selected value, all declared option probabilities,
token usage, timing, model revision, prompt version, and an explicit
uncalibrated-probability marker.

## Why FastJev?

The scoring primitive is intentionally simple: a causal model can expose
next-token logits. FastJev provides the application layer that a one-off
logprob call does not:

- **Zero-training decisions:** submit new criteria and option descriptions at
  runtime instead of collecting route examples or fine-tuning a classifier.
- **No JSON answer path:** the direct Torch and MLX backends read declared
  option logits and produce zero answer tokens—no schema repair, retry, or
  parsing loop.
- **A stable typed contract:** input validation, 2–16 option slots, normalized
  distributions, `Choice`/`Boolean`/`Score` values, and consistent errors.
- **Production boundaries:** resident Torch, batched vLLM, native MLX, a CLI,
  and an optional System One-compatible HTTP API share the same result model.
- **Auditable runs:** immutable model revisions, prompt hashes, row-level
  predictions, raw timings, and checksums are kept with the code.

If one binary prompt and raw logprobs are all you need, calling
[llama.cpp](https://github.com/ggml-org/llama.cpp) directly is simpler. FastJev
is useful when those decisions become an application surface that must stay
typed, portable, testable, and attributable.

## How it compares

These projects solve adjacent problems; this is a scope comparison, not a
cross-project speed benchmark.

| Project | Core mechanism | Choose it when | What FastJev does differently |
|---|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | Small, specialized non-autoregressive decision models | Low-resource or multilingual serving, especially when you can fine-tune for the workflow | Uses standard open causal models for zero-training, runtime-defined decisions and a 4,096-token default input limit |
| [semantic-router](https://github.com/aurelio-labs/semantic-router) | Embedding similarity against route utterances | Routes and example utterances are stable and vector similarity is enough | Evaluates a supplied state against new criteria and option meanings on every request |
| [Outlines](https://github.com/dottxt-ai/outlines) | Constrained autoregressive generation | You need arbitrary JSON, regex, grammar, or extraction schemas | Covers the narrower decision case and reads option scores without generating the full object |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Low-level local inference and token logprobs | You want maximum runtime control or a one-off classifier | Adds typed questions, prompt/slot validation, provenance, backends, HTTP compatibility, and frozen evaluations |

FastJev is not the best fit for every workload. Use a structured-generation
library for arbitrary extraction, an embedding router for a stable taxonomy, or
a trained small decision model when its domain and latency profile match your
deployment.

## Models you can run now

The same native BF16 direct-logit interface has been validated on these pinned
checkpoints. Put any listed model ID and revision into the quick start below:

| Model | Pinned source revision | Best use | Authored balanced accuracy | Browser option |
|---|---|---|---:|---:|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | Smallest starting point | 0.440 | Q8_0, 639 MB |
| [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | `12a3808a956f869c767195e9266b59c4d21d92e2` | Size/quality balance | 0.686 | Q4_K_M, 1.56 GB |
| **[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)** | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | **Recommended; best measured quality** | **0.813** | Q4_K_M, 3.01 GB |

The quality numbers are from native BF16 checkpoints, not the quantized browser
artifacts. Exact rows, revisions, and the separate browser smoke results are in
the [model-ladder report](results/raw/browser-model-ladder.json).

## Quick start

The default backend requires Python 3.10+, CUDA, and exactly one visible GPU.
Qwen3.5-4B uses about 9 GB of disk for the source checkpoint and measured
7.891 GiB peak allocated GPU memory in the historical
[RTX 5090 SDK smoke run](https://github.com/chengyongru/fastjev/pull/5).

```bash
git clone https://github.com/chengyongru/fastjev.git
cd fastjev
python -m venv .venv
. .venv/bin/activate
export HF_HOME=/path/to/large-drive/huggingface
pip install -e '.[torch]'
```

The first call downloads the pinned model revision from Hugging Face and caches
it under `HF_HOME`; no separate model download step is required.

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

Remote models require an immutable 40-character Hugging Face revision. Local
model directories are also supported and require a nonempty revision label for
result provenance.

Install `.[llama-cpp]` for local or Hugging Face-hosted GGUF files. The
[Python SDK guide](docs/SDK.md) covers llama.cpp setup and provenance.

## Measured results

### RTX 5090: Torch versus vLLM

A same-host integration run used the pinned Qwen3.5-4B checkpoint, identical
three-question inputs (125, 152, and 151 tokens), one warmup, and seven measured
`decide_many` calls:

| Backend | Model load | Median 3-decision batch | Decisions/s |
|---|---:|---:|---:|
| Torch | **10.04 s** | 161.97 ms | 18.52 |
| vLLM | 40.43 s | **82.73 ms** | **36.26** |

On this RTX 5090 / WSL workload, vLLM delivered 1.96× Torch throughput and
48.9% lower median batch latency. Its additional 30.38 seconds of startup cost
breaks even after roughly 383 three-decision batches when the process stays
resident. Both backends made the same three selections.

This is a historical integration measurement recorded in
[PR #7](https://github.com/chengyongru/fastjev/pull/7), not part of the
repository's reproducible benchmark bundle: the PR records the exact environment
and medians, but the seven individual timing samples were not committed.

### Decision quality

| Frozen workload | FastJev direct Qwen3.5-4B | Qwen3-Reranker-4B | Published Jev |
|---|---:|---:|---:|
| Authored decisions, 144 rows | **0.813** | 0.625 | — |
| WANLI, 256 rows | **0.637** | 0.522 | — |
| TypeSafe public subset, 102 rows / 20 cases | **0.845** | 0.560 | 0.883 |

The first two rows report balanced accuracy; the third reports equal-case modal
agreement. Jev was read from public records, not a live endpoint. See the full
[results, perturbations, and claim boundaries](docs/RESULTS.md).

## Backends and interfaces

| Runtime | Install | Best for |
|---|---|---|
| PyTorch/CUDA | `pip install -e '.[torch]'` | Default direct-logit scoring |
| vLLM/CUDA | `pip install -e '.[vllm]'` | Batched resident services |
| MLX/Apple Silicon | See the [MLX guide](docs/MLX.md) | Native macOS arm64 inference |
| WebGPU/GGUF | Open the [browser demo](webgpu-demo/index.html) | Local inference without Python |

Use `fastjev-score` for JSONL jobs. Install `.[api]` and run
`fastjev-serve` for `POST /v1/systemone` and `GET /v1/models`. The
[SDK guide](docs/SDK.md) covers batching and custom backends; the
[HTTP guide](docs/SYSTEM_ONE_API.md) covers server setup, authentication, and
compatibility boundaries.

## Documentation

- [Results](docs/RESULTS.md) — speed, quality, perturbations, and limitations
- [Method](docs/METHOD.md) — frozen prompts, metrics, and timing scope
- [Reproduce](docs/REPRODUCE.md) — pinned environments and verification commands
- [Benchmark bundle](benchmarks/README.md) — fixtures, runners, and source selection
- [Interactive replay](demo/index.html) and [browser-only WebGPU demo](webgpu-demo/index.html)

## Limits and provenance

FastJev's option probabilities are conditional on the supplied choices. They
are not calibrated confidence estimates; validate and calibrate them on the
deployment workload before using thresholds for consequential automation. The
experimental shared-prefix modes can change close BF16 argmax results.

Model weights and third-party evaluation records are not distributed here.
Upstream models retain their licenses; exact revisions are listed in
[THIRD_PARTY.md](THIRD_PARTY.md).

FastJev is an independently maintained fork of
[TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf), formerly OpenJev. It
preserves the original Git history and MIT license but follows an independent
roadmap. FastJev is not affiliated with or endorsed by TheoLeeCJ, TypeSafe, or
Jev, and it does not reproduce Jev's undisclosed model, training, calibration,
or performance.

Project code is released under the [MIT License](LICENSE).
