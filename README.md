<p align="center">
  <img src="./assets/fastjev-logo.svg"
       width="420"
       alt="FastJev" />
</p>

<div align="center">

[English](README.md) | [简体中文](README.zh-CN.md)

<img src="assets/fastjev-cover.webp" alt="FastJev self-hosted semantic decision SDK" width="100%">

**Fast, self-hosted semantic decisions with open models.**

</div>

FastJev is an open-source Python SDK for self-hosted semantic routing, LLM
classification, and structured AI decisions. Give it an unstructured state, a
runtime-defined question, and typed options; it returns option probabilities
without generating an answer sentence or parsing JSON.

Use it for routing, retry policies, evidence checks, and other narrow decisions
inside AI agents and applications.

## Why FastJev

- **Generation-free:** reads option logits directly, with no decoding loop or
  generated output to repair.
- **Typed Python API:** supports `Choice`, `Boolean`, and `Score` results with
  stable option IDs and provenance metadata.
- **Self-hosted open models:** runs through PyTorch/CUDA, vLLM, or MLX on Apple
  Silicon.
- **Built for repeated decisions:** keeps the model resident and supports batch
  scoring and shared-state prefix reuse.
- **Auditable:** pins model revisions and publishes prompts, row-level outputs,
  benchmark runners, and checksums.
- **API-compatible:** optionally serves the documented System One request shape
  over HTTP.

## Quick start

The default backend requires Python 3.10+, CUDA, and one GPU that can hold a 4B
BF16 model:

```bash
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
        state="Customer cannot access an account after a password reset.",
        question=Choice("Which queue should handle this request?", [
            Option("access", "Account access support."),
            Option("billing", "Billing support."),
        ]),
    )

print(result.value)
print(result.probabilities)
```

Remote models require an immutable 40-character Hugging Face revision. Local
model directories are also supported and require a nonempty revision label for
result provenance.

Install `.[llama-cpp]` for local or Hugging Face-hosted GGUF files. The
[Python SDK guide](docs/SDK.md) covers llama.cpp setup and provenance.

## Backends

| Runtime | Install | Best for |
|---|---|---|
| PyTorch/CUDA | `pip install -e '.[torch]'` | Default direct-logit scoring |
| vLLM/CUDA | `pip install -e '.[vllm]'` | Batched resident inference |
| MLX/Apple Silicon | See the [MLX guide](docs/MLX.md) | Native macOS arm64 inference |

All backends implement the same `ScoringBackend` protocol, so application code
can keep the same typed decisions and result objects when the runtime changes.
See the [Python SDK guide](docs/SDK.md) for batching, backend injection, result
semantics, and lifecycle management.

For local browser inference, the [WebGPU demo](webgpu-demo/index.html) downloads
a pinned GGUF model on demand and stores it in browser-managed cache.

## CLI and HTTP API

Score JSONL from the command line:

```bash
CUDA_VISIBLE_DEVICES=0 fastjev-score \
  --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl \
  --output results.jsonl
```

Use `--mode shared` when every input row has the same state and its criteria can
reuse one prefix. For a resident service, install `.[api]` and run
`fastjev-serve`; the [HTTP API guide](docs/SYSTEM_ONE_API.md) documents
`POST /v1/systemone`, authentication, and compatibility boundaries.

## Measured performance

On one RTX 3090, the frozen Qwen3.5-4B benchmark evaluated the same state and 21
binary criteria:

| Output path | Median time | Output tokens |
|---|---:|---:|
| Direct typed logits | **1.023 s** | **0** |
| Autoregressive JSON array | 5.332 s | 111 |

The generated baseline emitted only an ordered array of `"yes"`/`"no"` values.
It took 5.21× as long to complete, while its choices agreed with direct argmax on
18 of 21 criteria. This measures output-path cost; it does not claim that the two
readouts are semantically equivalent. See the [raw run](results/raw/decision-vs-compact-array.json)
and the full [results and limitations](docs/RESULTS.md).

## Documentation

- [Python SDK](docs/SDK.md) — typed decisions, batching, and backend contracts
- [System One-compatible API](docs/SYSTEM_ONE_API.md) — server setup and wire format
- [MLX backend](docs/MLX.md) — Apple Silicon setup and cache behavior
- [Results](docs/RESULTS.md) — speed, quality, perturbations, and claim boundaries
- [Method](docs/METHOD.md) — frozen prompts, metrics, and timing scope
- [Reproduce](docs/REPRODUCE.md) — pinned environments and verification commands
- [Benchmark bundle](benchmarks/README.md) — fixtures, runners, and source selection
- [Interactive replay](demo/index.html) and [browser-only WebGPU demo](webgpu-demo/index.html)

## Limitations and provenance

FastJev returns probabilities conditional on the supplied options. They are not
calibrated confidence estimates; validate and calibrate them on the workload
where they will make decisions. Fast prefix-reuse paths can also change BF16
argmax results and remain experimental.

Model weights and third-party evaluation records are not distributed in this
repository. Upstream models retain their own licenses and exact revisions are
listed in [THIRD_PARTY.md](THIRD_PARTY.md).

FastJev is an independently maintained fork of
[TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf), formerly OpenJev. It
preserves the original Git history and MIT license but follows an independent
roadmap. FastJev is not affiliated with or endorsed by TheoLeeCJ, TypeSafe, or
Jev, and it does not reproduce Jev's undisclosed model, training, calibration,
or performance.

Project code is released under the [MIT License](LICENSE).
