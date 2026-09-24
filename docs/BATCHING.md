# Cross-state batching measurements

[English](BATCHING.md) | [简体中文](BATCHING.zh-CN.md)

`FastJev.decide_batch(states, questions)` applies one question mapping to multiple
states. Torch tensor batching is opt-in through `batch_size`; the default remains
sequential. See the [SDK guide](SDK.md#evaluate-multiple-states) for ordering,
memory, timing, and probability semantics.

## Mixed-length public SDK workload

The [raw report](../results/raw/runtime/rtx5090-cross-state-batching.json) records
24 owned states, three typed questions per state, one warmup per mode, and five
complete calls per mode. All 72 decisions are retained for every measured call.
States mix English and Chinese messages with varying amounts of unrelated context.
They are a throughput and parity fixture, not an accuracy benchmark.

All modes share one loaded Qwen3.5-4B BF16 checkpoint at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, on one RTX 5090 under WSL2, with
PyTorch 2.10.0+cu128 and Transformers 5.17.0. The optional causal-conv1d and
flash-linear-attention kernels were absent. Timings include public SDK validation,
encoding, inference, and result conversion, exclude loading and warmup, and
synchronize CUDA. Modes run sequentially: serial, batch 8, then length grouping.

| Execution | Median call | Decisions/s | Peak allocated GiB | Changed selections | Maximum probability drift |
|---|---:|---:|---:|---:|---:|
| Sequential | 4.666 s | 15.43 | 8.05 | reference | 0 |
| Batch 8 | 3.991 s | 18.04 | 9.53 | 0/360 | 0.037524 |
| Batch 8, length grouping | 2.975 s | 24.20 | 9.53 | 0/360 | 0.030953 |

Length grouping achieved 1.57× the sequential throughput on this workload. The
360 comparisons cover five repetitions of 72 decisions, not 360 independent
examples. Unchanged selections do not imply unchanged probabilities or validated
calibration. Peak allocation includes model tensors and is not total process VRAM.

## Frozen quality fixtures and this-that comparison

The [model comparison](../results/raw/runtime/rtx5090-cross-state-model-comparison-verified.json)
uses the same hardware and precision, all 144 authored decisions and 108 owned
perturbations. Each fixture contains 36 source groups; perturbations derive from
the authored originals and are not an independent corpus. Questions, option order,
and evidence are preserved. Each mode has eight warmup rows and one measured pass
per fixture. No model generates output tokens or truncates these inputs.

The cached `flock-io/this-that-model-1.0` checkpoint is pinned to
`3d927195c4f9845efe66c5715883a7a0f42b1239`; the runner checks the weight SHA-256.
Its native path uses `thisthat` source revision
`542d445efa5f68b14bfbd1f8ed25aacd8379d839`, `state_first`, temperature 1, one
question per call, and an explicitly checked 4,096-token limit. The generic path
uses FastJev's unchanged `direct-options-v1` prompt and full-vocabulary readout.
This compares complete inference paths, not architecture alone.

| Model / execution | Authored mean-family balanced accuracy | Perturbation mean-family balanced accuracy |
|---|---:|---:|
| Qwen3.5-4B, sequential | 0.8132 | 0.7720 |
| Qwen3.5-4B, batch 8 + length grouping | 0.8132 | 0.7640 |
| this-that, FastJev sequential | 0.8660 | 0.7483 |
| this-that, FastJev batch 8 + length grouping | 0.8676 | 0.7434 |
| this-that, native sequential | 0.8286 | 0.8556 |

Qwen batching preserves all authored selections but changes one perturbation
selection, reducing accuracy from 83/108 to 82/108. This is why batching remains
an explicit deployment choice. The smaller this-that model is a useful candidate:
sequential peak allocation is about 3.5 GiB versus 7.9 GiB for Qwen in this fixture.
Its ranking depends on the prompt path and workload. All paired source-group
bootstrap 95% intervals comparing this-that against sequential Qwen include zero;
these samples do not establish a general quality advantage.

Confidence also needs separate validation: authored ECE is 0.0567 for Qwen,
0.0855 for generic this-that, and 0.0917 for native this-that. Native this-that's
perturbation ECE is 0.0708 versus Qwen's 0.1256. No temperature was fitted to these
fixtures. Single-pass quality timings in the raw report are diagnostic; the
five-repeat SDK workload above is the dedicated throughput measurement.

## Reproduce

Run from the repository root in an isolated environment installed with
`pip install -e '.[test,torch]'`. Each scorer process must see exactly one GPU and
each output path must be new.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python benchmarks/cross_state_batching.py \
  --output /path/to/new-batching-report.json

# Optional dependency for the native this-that comparison only.
pip install --no-deps \
  'thisthat @ git+https://github.com/FLock-io/this-that-model@542d445efa5f68b14bfbd1f8ed25aacd8379d839'
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python benchmarks/compare_decision_models.py \
  --flock /path/to/this-that-model-1.0-3d927195 \
  --output /path/to/new-model-comparison.json

(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```

`--flock` must contain the complete pinned model and tokenizer files downloaded
from Hugging Face. Models and caches are external to this repository. Reports
record exact inputs or owned fixture hashes, model revisions, implementation
hashes, per-decision predictions, timing, and memory. The verifier recomputes batch
timing/drift and fixture accuracy from the committed predictions.
