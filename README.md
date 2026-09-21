<p align="center">
  <img src="./assets/fastjev-logo.svg"
       width="420"
       alt="FastJev" />
</p>

<div align="center">

[English](README.md) | [简体中文](README.zh-CN.md)

<img src="assets/fastjev-cover.webp" alt="FastJev open source Jev implementation for self deployment" width="100%">

**Deploy an open source Jev implementation on your own infrastructure.**

</div>

FastJev is an open source implementation of
[Jev](https://docs.typesafe.ai/) for deployment on infrastructure you control.
It continues [SemIf](https://github.com/TheoLeeCJ/SemIf) as an independently
maintained fork. Pinned open models run the `Choice`, `Boolean`, and `Score`
interface through Torch, vLLM, MLX, and llama.cpp. The WebGPU demo provides
browser local inference.

## Why FastJev?

FastJev turns Jev deployment into a standard Python workflow. The SDK loads the
model, validates 2 to 16 options, performs scoring, and returns typed results
with probabilities, token usage, timing, model revision, and prompt version.

Resident Torch, batched vLLM, native MLX, llama.cpp GGUF, the CLI, and the
optional System One compatible HTTP API share the same result model. Torch,
MLX, and llama.cpp return results in one scoring pass with zero output tokens.

Each result records the model revision and prompt version. Published
evaluations add row data, raw timings, and checksums for reproducibility.

## Models you can run now

The same native BF16 direct logit interface has been validated on these pinned
checkpoints. Put any listed model ID and revision into the quick start below.

| Model | Pinned source revision | Best use | Authored balanced accuracy | Browser option |
|---|---|---|---:|---:|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | Smallest starting point | 0.440 | Q8_0, 639 MB |
| [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | `12a3808a956f869c767195e9266b59c4d21d92e2` | Size/quality balance | 0.686 | Q4_K_M, 1.56 GB |
| **[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)** | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | **Recommended for highest measured quality** | **0.813** | Q4_K_M, 3.01 GB |

The table reports native BF16 checkpoint quality. Browser artifacts use
separate quantized formats. Their smoke results, exact rows, and revisions are
in the [model ladder report](results/raw/browser-model-ladder.json).

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

The first call automatically downloads the pinned model revision from Hugging
Face and caches it under `HF_HOME`.

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

Remote models use an immutable Hugging Face revision containing 40 characters.
Local model directories use a descriptive revision label for result provenance.

Install `.[llama-cpp]` to run GGUF files from disk or Hugging Face. The
[Python SDK guide](docs/SDK.md) covers llama.cpp setup and provenance.

## Measured results

### RTX 5090 with Torch and vLLM

An integration run on one host used the pinned Qwen3.5-4B checkpoint, identical
inputs for three questions (125, 152, and 151 tokens), one warmup, and seven
measured `decide_many` calls.

| Backend | Model load | Median batch of three decisions | Decisions/s |
|---|---:|---:|---:|
| Torch | **10.04 s** | 161.97 ms | 18.52 |
| vLLM | 40.43 s | **82.73 ms** | **36.26** |

On this RTX 5090 / WSL workload, vLLM delivered 1.96× Torch throughput and
48.9% lower median batch latency. Its additional 30.38 seconds of startup cost
breaks even after roughly 383 batches of three decisions when the process stays
resident. Both backends made the same three selections.

[PR #7](https://github.com/chengyongru/fastjev/pull/7) records this historical
integration measurement, including the exact environment and aggregate medians.
The repository's reproducible benchmark bundle covers separate experiments with
committed row data.

### Decision quality

| Frozen workload | FastJev direct Qwen3.5-4B | Qwen3-Reranker-4B | Published Jev |
|---|---:|---:|---:|
| Authored decisions, 144 rows | **0.813** | 0.625 | N/A |
| WANLI, 256 rows | **0.637** | 0.522 | N/A |
| TypeSafe public subset, 102 rows / 20 cases | **0.845** | 0.560 | 0.883 |

The first two rows report balanced accuracy. The third reports modal agreement
with equal weighting across cases. The Jev column reproduces public records.
The two open model columns come from local frozen evaluations. The
[results report](docs/RESULTS.md) provides the full data, perturbations, and
claim boundaries.

## How it compares

These projects serve adjacent deployment needs. Their published performance
belongs to their own workloads.

| Project | Core mechanism | Choose it when | FastJev focus |
|---|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | Small decision models with parallel option scoring | Serving with limited resources or multiple languages, especially when fine tuning matches the workflow | Applies standard open causal models to decisions defined at runtime, with pinned revisions and a 4,096 token default input limit |
| [semantic-router](https://github.com/aurelio-labs/semantic-router) | Embedding similarity against route utterances | Routes and example utterances are stable and vector similarity is enough | Evaluates a supplied state against new criteria and option meanings on every request |
| [Outlines](https://github.com/dottxt-ai/outlines) | Constrained autoregressive generation | You need arbitrary JSON, regex, grammar, or extraction schemas | Focuses on typed decisions and reads option scores in one scoring pass |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Local inference and token logprobs | You want maximum runtime control or a single classifier | Uses llama.cpp as a backend and adds typed questions, validation, provenance, multiple runtimes, HTTP compatibility, and frozen evaluations |

## Backends and interfaces

| Runtime | Install | Best for |
|---|---|---|
| PyTorch/CUDA | `pip install -e '.[torch]'` | Default scoring from logits |
| vLLM/CUDA | `pip install -e '.[vllm]'` | Batched resident services |
| MLX/Apple Silicon | See the [MLX guide](docs/MLX.md) | Native macOS arm64 inference |
| llama.cpp/GGUF | `pip install -e '.[llama-cpp]'` | GGUF files from disk or Hugging Face |
| WebGPU/GGUF | Open the [browser demo](webgpu-demo/index.html) | Inference in the browser |

Use `fastjev-score` for JSONL jobs. Install `.[api]` and run
`fastjev-serve` for `POST /v1/systemone` and `GET /v1/models`. The
[SDK guide](docs/SDK.md) covers batching and custom backends. The
[HTTP guide](docs/SYSTEM_ONE_API.md) covers server setup, authentication, and
compatibility boundaries.

## Documentation

The [results report](docs/RESULTS.md) covers speed, quality, perturbations, and
limitations. The [method guide](docs/METHOD.md) documents frozen prompts,
metrics, and timing scope. The [reproduction guide](docs/REPRODUCE.md) provides
pinned environments and verification commands. The
[benchmark bundle](benchmarks/README.md) contains fixtures, runners, and source
selection. The [interactive replay](demo/index.html) and
[WebGPU browser demo](webgpu-demo/index.html) provide visual ways to explore
the project.

## Boundaries and provenance

FastJev returns probabilities conditioned on the supplied options with
`calibrated=False`.
Deployment validation and calibration establish thresholds for consequential
automation. The experimental shared prefix modes can change close BF16 argmax
results.

Model weights stay on upstream hosts, and third party evaluation records stay
with their original sources. Upstream models retain their licenses. Exact
revisions are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

The [third party record](THIRD_PARTY.md) documents source and model provenance.

Project code is released under the [MIT License](LICENSE).
