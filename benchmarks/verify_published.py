"""Verify that the machine-readable summary is backed by committed raw evidence."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with (ROOT / path).open() as stream:
        return json.load(stream)


def close(left, right, tolerance=5e-10):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if abs(left - right) > tolerance:
            raise AssertionError(f"{left!r} != {right!r}")
    elif left != right:
        raise AssertionError(f"{left!r} != {right!r}")


def top_choice(row):
    return row["option_ids"][max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__)]


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def main():
    summary = load("results/phase1-summary.json")
    quality = load("results/raw/quality-comparison.json")
    perturb = load("results/raw/perturbation-comparison.json")
    direct_shape = load("results/raw/shape777-direct.json")
    reranker_shape = load("results/raw/shape777-reranker.json")
    compact = load("results/raw/decision-vs-compact-array.json")
    gguf_authored = load("results/raw/llama-cpp-qwen3.5-4b-q4-k-m-authored144.json")
    gguf_perturbations = load("results/raw/llama-cpp-qwen3.5-4b-q4-k-m-perturbations108.json")
    gguf_sdk = load("results/raw/llama-cpp-qwen3.5-4b-q4-k-m-rtx5090-sdk.json")
    checks = 0

    semantic = summary["semantic_quality"]
    claims = {
        "authored_144_mean_family_balanced_accuracy": (
            quality["hard_label"]["authored"]["direct_logits"]["mean_family_balanced_accuracy"],
            quality["hard_label"]["authored"]["reranker"]["mean_family_balanced_accuracy"],
        ),
        "wanli_256_balanced_accuracy": (
            quality["hard_label"]["wanli"]["direct_logits"]["mean_family_balanced_accuracy"],
            quality["hard_label"]["wanli"]["reranker"]["mean_family_balanced_accuracy"],
        ),
        "every_judge_grid_36_accuracy": (
            quality["every"]["direct_logits"]["judge-grid"]["accuracy"],
            quality["every"]["reranker"]["judge-grid"]["accuracy"],
        ),
        "every_action_firewall_10_accuracy": (
            quality["every"]["direct_logits"]["action-firewall"]["accuracy"],
            quality["every"]["reranker"]["action-firewall"]["accuracy"],
        ),
        "every_code_rag_recall_at_1": (
            quality["every"]["direct_logits"]["code-rag"]["recall_at_1"],
            quality["every"]["reranker"]["code-rag"]["recall_at_1"],
        ),
        "every_company_brain_recall_at_1": (
            quality["every"]["direct_logits"]["company-brain"]["recall_at_1"],
            quality["every"]["reranker"]["company-brain"]["recall_at_1"],
        ),
    }
    for key, (direct, reranker) in claims.items():
        close(semantic[key]["direct_logits"], direct)
        close(semantic[key]["reranker"], reranker)
        checks += 2
    typesafe = quality["typesafe"]["systems"]
    for suffix, field in (("modal_agreement", "agreement"), ("tv_distance", "tv")):
        claim = semantic[f"typesafe_public_102_equal_case_{suffix}" if suffix == "modal_agreement" else "typesafe_public_102_tv_distance"]
        close(claim["direct_logits"], typesafe["direct_logits"][field])
        close(claim["reranker"], typesafe["reranker"][field])
        close(claim["published_jev"], typesafe["typesafe"][field])
        checks += 3

    for system in ("direct_logits", "reranker"):
        source = perturb["systems"][system]
        claim = summary["perturbations_36"][system]
        close(claim["base_balanced_accuracy"], source["base_original"]["mean_family_balanced_accuracy"])
        for variant in ("option_reversal", "criterion_wrapper", "irrelevant_context"):
            close(claim[variant]["balanced_accuracy"], source["variants"][variant]["evaluation"]["mean_family_balanced_accuracy"])
            close(claim[variant]["argmax_flips"], source["variants"][variant]["argmax_flips"])
            checks += 2
        close(claim["missing_evidence_confident_non_insufficient_at_0_8"], source["missing_evidence"]["confident_non_insufficient_at_0_8"])
        checks += 2

    direct_comparisons = {"fresh_batch1": 0, "serial_prefix": 5, "parallel_suffix": 6}
    for claim, raw in zip(summary["shape777"]["direct"], direct_shape["results"]):
        for summary_key, raw_key in (("wall_seconds", "wall_seconds"), ("judgments_per_second", "judgments_per_second"),
                                     ("state_p50_seconds", "state_latency_p50_seconds"), ("peak_cuda_bytes", "peak_cuda_bytes")):
            close(claim[summary_key], raw[raw_key])
            checks += 1
        close(claim["argmax_flips_vs_fresh"], direct_comparisons[claim["mode"]])
        checks += 1

    reranker_predictions = defaultdict(dict)
    for row in rows("results/raw/shape777-reranker.predictions.jsonl"):
        reranker_predictions[row["pair_batch_size"]][row["id"]] = top_choice(row)
    reference = reranker_predictions[1]
    for claim, raw in zip(summary["shape777"]["reranker"], reranker_shape["results"]):
        for summary_key, raw_key in (("wall_seconds", "wall_seconds"), ("judgments_per_second", "judgments_per_second"),
                                     ("state_p50_seconds", "state_latency_p50_seconds"), ("peak_cuda_bytes", "peak_cuda_bytes")):
            close(claim[summary_key], raw[raw_key])
            checks += 1
        flips = sum(choice != reference[row_id] for row_id, choice in reranker_predictions[claim["pair_batch_size"]].items())
        close(claim["argmax_flips_vs_batch1"], flips)
        checks += 1

    generation = summary["decision_vs_compact_generation21"]
    close(generation["direct_parallel"]["median_seconds"], compact["direct_parallel"]["median_total_seconds"])
    close(generation["compact_generation"]["median_seconds"], compact["compact_generation"]["median_total_seconds"])
    close(generation["compact_generation"]["median_output_tokens"], compact["compact_generation"]["median_output_tokens"])
    close(generation["compact_generation"]["agreement_with_direct_argmax"], compact["compact_generation"]["agreement_with_direct_argmax_first_run"])
    close(generation["wall_time_ratio_generation_over_direct"], compact["median_wall_ratio"])
    checks += 5

    gguf_sets = (
        (
            "authored",
            rows("benchmarks/data/authored144.jsonl"),
            rows("results/raw/predictions/llama-cpp-qwen3.5-4b-q4-k-m-authored144.jsonl"),
            rows("results/raw/predictions/direct-authored144.jsonl"),
            gguf_authored,
            144,
            138,
            0.803,
            0.813,
        ),
        (
            "perturbations",
            rows("benchmarks/data/perturbations108.jsonl"),
            rows("results/raw/predictions/llama-cpp-qwen3.5-4b-q4-k-m-perturbations108.jsonl"),
            rows("results/raw/predictions/direct-perturbations108.jsonl"),
            gguf_perturbations,
            108,
            105,
            0.775,
            0.780,
        ),
    )
    artifact_sha256 = "13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983"
    artifact_revision = "4168f45a16a1290d65a4ec0fa312ae917a4c15d6"
    for name, gold_rows, gguf_rows, bf16_rows, report, count, agreement, gguf_claim, bf16_claim in gguf_sets:
        gold_ids = {row["id"] for row in gold_rows}
        gguf_by_id = {row["id"]: row for row in gguf_rows}
        bf16_by_id = {row["id"]: row for row in bf16_rows}
        if len(gold_ids) != count or set(gguf_by_id) != gold_ids or set(bf16_by_id) != gold_ids:
            raise AssertionError(f"Incomplete llama.cpp {name} evidence")
        if any(
            row["output_tokens"] != 0
            or row["model"]["revision"] != artifact_revision
            or row["model"]["source_artifact_sha256"] != artifact_sha256
            or row["model"]["n_gpu_layers"] != -1
            for row in gguf_rows
        ):
            raise AssertionError(f"Invalid llama.cpp {name} provenance")
        observed_agreement = sum(
            top_choice(gguf_by_id[row_id]) == top_choice(bf16_by_id[row_id])
            for row_id in gold_ids
        )
        close(observed_agreement, agreement)
        close(report["coverage"], 1.0)
        if round(report["mean_family_balanced_accuracy"], 3) != gguf_claim:
            raise AssertionError(f"Unexpected rounded llama.cpp {name} accuracy")
        bf16_accuracy = report["mean_family_balanced_accuracy"] - report["paired_comparison"]["difference"]
        if round(bf16_accuracy, 3) != bf16_claim:
            raise AssertionError(f"Unexpected rounded BF16 {name} accuracy")
        checks += 4

    if gguf_sdk["model"]["sha256"] != artifact_sha256 or gguf_sdk["model"]["revision"] != artifact_revision:
        raise AssertionError("SDK benchmark model provenance disagrees")
    runtime = gguf_sdk["runtime"]
    if runtime["llama_cpp_python"] != "0.3.35" or not runtime["supports_gpu_offload"] or runtime["n_gpu_layers"] != -1:
        raise AssertionError("SDK benchmark did not verify CUDA GPU offload")
    measurements = gguf_sdk["measurements"]
    median = statistics.median(row["wall_seconds"] for row in measurements)
    close(gguf_sdk["summary"]["median_batch_seconds"], median)
    close(gguf_sdk["summary"]["decisions_per_second_at_median"], 3 / median)
    if len(gguf_sdk["warmups"]) != 1 or len(measurements) != 7:
        raise AssertionError("Unexpected SDK benchmark repeat counts")
    if not gguf_sdk["summary"]["all_expected_selections"] or not gguf_sdk["summary"]["all_output_tokens_zero"]:
        raise AssertionError("SDK safeguard smoke failed")
    if round(gguf_sdk["load_seconds"], 2) != 4.05 or round(gguf_sdk["summary"]["median_batch_seconds"], 3) != 2.344:
        raise AssertionError("SDK README timing claims disagree")
    gpu = gguf_sdk["gpu"]
    if gpu["before_load"]["gpus"][0]["memory_used_mib"] != 82 or gpu["after_load"]["gpus"][0]["memory_used_mib"] != 4003:
        raise AssertionError("SDK README GPU-memory claims disagree")
    checks += 8
    print(json.dumps({"verified_summary_claims": checks, "status": "ok"}))


if __name__ == "__main__":
    main()
