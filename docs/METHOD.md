# Method

## Question and systems

Phase 1 tests whether open, generation-free readouts reproduce the useful part of Jev's public claim: accept unstructured state plus runtime-defined natural-language decisions and return typed scores cheaply enough to embed in ordinary software.

The direct system prompts frozen Qwen3.5-4B with a state, criterion, and 2-16 described options. It performs one native forward pass and applies a softmax only to the logits of fixed uppercase answer tokens. It does not decode a token.

The reranker system follows Qwen3-Reranker-4B's native yes/no contract. Each candidate answer becomes a separate query/document relevance proposition. The system computes `logit(yes) - logit(no)` for every option and softmaxes those log-odds across options. That last normalization is our comparison rule; it is not part of the upstream reranker's calibration contract.

## Frozen evaluation matrix

Prompts, IDs, labels, task semantics, revisions, and metrics were frozen before the full reranker outputs were evaluated. The complete local matrix had 706 rows:

| Source | Rows | Purpose |
|---|---:|---|
| Project-authored | 144 | Evidence interpretation, rule application, candidate selection; original and missing-evidence cases |
| WANLI | 256 | External natural-language inference |
| TypeSafe public evaluation subset | 102 | Distribution/reference agreement on 20 available cases |
| Every public lab artifacts | 204 | Judgment grid, retrieval, company knowledge, composed action policy |

Unlike tasks were not collapsed into one accuracy number. Hard-label tasks use full-denominator accuracy, balanced accuracy, macro F1, NLL/Brier where applicable, and source-group bootstrap intervals. Retrieval reports query ranking metrics. TypeSafe rows compare distributions and use an equal-case macro so cases with more questions do not dominate.

The TypeSafe comparison is the 102 public rows that could be aligned locally, not its advertised 711-row aggregate and not a live Jev run. Published Jev, Opus, and Sol values were read from those public records. Raw third-party fixtures are excluded from this candidate.

The exact evaluated IDs are committed in `benchmarks/manifests/source-selection.jsonl`. WANLI uses revision `61c95318fd71c55b6ba355d76253254615f387ec`: malformed or over-4,000-character rows and components touching pilot-training sources were excluded, remaining IDs were sorted then shuffled with seed 291607, and 86 entailment/85 contradiction/85 neutral rows were selected with at most one row per connected premise/pair-ID component. Premise becomes state; hypothesis becomes the criterion; entailment/neutral/contradiction map to supported/insufficient/contradicted.

TypeSafe extraction reads four locally supplied, hash-verified `*-cases.js` snapshots. It keeps published, successfully run Choice/Noul nodes having one unambiguous document/question binding, a released reference answer, and a unique reference-distribution argmax. Selection is outcome-blind round-robin over workflow, case, and primitive, with hashed-ID order inside buckets. Score primitives and tied targets are excluded from the 102-row comparison. Every mappings and original experiment data are available from the directly linked experiment JSON and source archive.

## Perturbations

Thirty-six owned original cases received three output-blind variants: reverse the displayed option order while preserving semantic IDs, wrap the criterion in meaning-preserving wording, and append irrelevant owned context. A separate 36-row missing-evidence population tests whether a system selects `insufficient`. Stability is measured after aligning probabilities by semantic option ID.

## Browser model ladder

Qwen3-0.6B, MiniCPM5-2B, and Qwen3.5-4B use the same frozen prompt and native BF16 final-position option-logit scorer on the 144 authored, 108 perturbation, and 102 selected TypeSafe rows. TypeSafe modal agreement is averaged within each of the 20 source cases and then equally across cases. The browser artifacts are independently pinned GGUF quantizations. Browser smoke timings begin after the page initiates each operation; model files were served from a local SSD to exclude internet transfer time. A successful smoke requires model load, warmup, finite logits for every displayed option, and completion of the generated path. It does not establish full quantized quality or portable latency.

## llama.cpp GGUF validation

The server-side GGUF validation used
`bartowski/Qwen_Qwen3.5-4B-GGUF` revision
`4168f45a16a1290d65a4ec0fa312ae917a4c15d6`, exact file
`Qwen_Qwen3.5-4B-Q4_K_M.gguf`, and SHA-256
`13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983`.
FastJev 0.1.1 at commit `e9ee737` ran through `llama-cpp-python` 0.3.35's
CUDA 13.0 wheel with `n_gpu_layers=-1`, `n_batch=512`, and one visible RTX
5090 under WSL2. The runtime GPU-offload probe had to return true.

Quality uses the same prompt contract and evaluator as the primary committed
Torch BF16 predictions. llama.cpp direct mode scored all 144 authored and 108
perturbation rows independently; missing or invalid rows would remain failures.
The paired BF16 files use the native state-prefix cache. The reports compare
semantic-ID-aligned argmax results across these complete runtime paths, so they
do not isolate quantization from serving-shape and kernel effects.

`benchmarks/llama_cpp_sdk.py` defines a separate three-question shell-safeguard
smoke. No command is executed. SDK construction time includes GGUF loading and
the provenance SHA-256 pass. After one unmeasured warmup, seven complete
`decide_many` calls are timed around the public API. Each call contains the same
three questions, which llama.cpp currently evaluates in order. Model transfer,
process startup, result serialization, and the warmup are outside the reported
median. GPU memory values are whole-device `nvidia-smi` observations, not
allocator-only measurements.

## Shape-matched systems benchmark

An owned fixture contains 37 states and 21 fixed binary criteria per state, giving 777 decisions. States are roughly 8,000 characters and exercise repeated-context computation. It matches the count geometry of the public Every/Jev demonstration, but does not reproduce its unpublished documents, token lengths, hardware, API path, or model. Therefore it is a systems measurement, not a Jev head-to-head benchmark.

Direct modes are fresh batch-one scoring, serial suffixes after one state prefill, and parallel suffix branches after one state prefill. The reranker repeats the state for two independent yes/no option pairs per binary decision and tests ordinary pair batching. Timings use one RTX 3090 with a warm-loaded BF16 model and include prompt construction, tokenization, transfers, forward passes, and CPU readout; model loading and result-file writes are outside the timed region.

## Interpretation rules

- A forced typed output can still be semantically wrong.
- Softmax over allowed tokens is conditional on the supplied alternatives; it is not calibrated operational confidence.
- Prefix-cache speedups are implementation results, not evidence about Jev's disclosed architecture.
- A reranker is expected to be strongest on ranking. Its categorical threshold metrics should not be confused with ranking quality.
