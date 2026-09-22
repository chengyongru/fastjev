"""Create-only Apple Silicon evidence: quality, cache reuse, and compact generation.

Run from the repository root. Reuses the frozen fixtures and evaluator; does not
modify the CUDA evidence. Each run requires a new output directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import evaluate
from mlx_evidence import read_bytes
from decision_vs_generation import compact_messages
from fastjev._runtime import mlx as backend
from fastjev._runtime.direct import encode_prompt


def read(path):
    return [json.loads(line) for line in read_bytes(path).decode("utf-8").splitlines() if line.strip()]


def write(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def choice(row):
    return row["option_ids"][max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__)]


def compare(reference, candidates):
    old = evaluate.indexed(reference, "reference")
    new = evaluate.indexed(candidates, "candidate")
    if old.keys() != new.keys():
        raise ValueError("Comparison requires identical decision IDs")
    differences, flips, prompt_mismatches = [], [], []
    for row in candidates:
        base = old[row["id"]]
        left = dict(zip(base["option_ids"], base["probabilities"]))
        right = dict(zip(row["option_ids"], row["probabilities"]))
        if left.keys() != right.keys():
            raise ValueError("Option IDs differ")
        differences.append(max(abs(left[key] - right[key]) for key in left))
        if choice(row) != choice(base):
            flips.append({"id": row["id"], "from": choice(base), "to": choice(row),
                          "reference_probabilities": left, "candidate_probabilities": right})
        if row["prompt_sha256"] != base["prompt_sha256"] or row["input_tokens"] != base["input_tokens"]:
            prompt_mismatches.append(row["id"])
    return {"rows": len(candidates), "argmax_flips": flips, "prompt_mismatches": prompt_mismatches,
            "max_probability_difference": max(differences),
            "mean_max_probability_difference": statistics.mean(differences)}


def quality(model, tokenizer, metadata, output, reference_run=None):
    import mlx.core as mx

    datasets = {}
    report = {}
    for name in ("authored144", "perturbations108"):
        gold = read(f"benchmarks/data/{name}.jsonl")
        backend.score(model, tokenizer, gold[0], metadata)
        mx.reset_peak_memory()
        started = time.perf_counter()
        predictions = []
        for index, row in enumerate(gold):
            predictions.append(backend.score(model, tokenizer, row, metadata))
            if (index + 1) % 36 == 0:
                print(f"{name}: {index + 1}/{len(gold)}", flush=True)
        elapsed = time.perf_counter() - started
        write_rows(output / f"{name}.jsonl", predictions)
        baseline = read(f"results/raw/predictions/direct-{name}.jsonl")
        report[name] = {"evaluation": evaluate.evaluate(gold, predictions, comparison=baseline),
                        "vs_published_torch": compare(baseline, predictions),
                        "wall_seconds": elapsed, "peak_mlx_bytes": mx.get_peak_memory()}
        if reference_run is not None:
            reference = read(reference_run / f"{name}.jsonl")
            report[name]["vs_mlx_reference"] = compare(reference, predictions)
        datasets[name] = (gold, predictions)
    gold, base = datasets["authored144"]
    perturb_gold, perturb = datasets["perturbations108"]
    by_id = {row["id"]: row for row in base}
    variants = defaultdict(list)
    for row, prediction in zip(perturb_gold, perturb):
        variants[row["provenance"]["variant"]].append((row, prediction))
    report["perturbation_stability"] = {}
    for name, pairs in variants.items():
        references = [dict(by_id[row["provenance"]["base_id"]], id=row["id"]) for row, _ in pairs]
        comparison = compare(references, [pred for _, pred in pairs])
        # Wording/order/context intentionally change the prompts in this comparison.
        comparison.pop("prompt_mismatches")
        report["perturbation_stability"][name] = comparison
    missing = [row for row in gold if row["provenance"]["variant"] == "missing"]
    report["missing_evidence_confident_non_insufficient_at_0_8"] = sum(
        choice(by_id[row["id"]]) != "insufficient" and max(by_id[row["id"]]["probabilities"]) >= .8
        for row in missing
    )
    write(output / "quality.json", report)
    return {name: report[name]["evaluation"]["mean_family_balanced_accuracy"] for name in datasets}


def shape(model, tokenizer, metadata, output):
    import mlx.core as mx

    rows = read("benchmarks/data/shape777.jsonl")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(rows) != 777 or len(groups) != 37 or any(len(group) != 21 for group in groups.values()):
        raise ValueError("Expected 37 groups of 21 decisions")
    first = next(iter(groups.values()))
    predictions, reports = {}, []
    for mode in ("direct", "serial", "shared"):
        def run(group):
            if mode == "direct":
                return [backend.score(model, tokenizer, row, metadata) for row in group]
            if mode == "serial":
                scorer = backend.SerialPrefixScorer(model, tokenizer, metadata)
                return [scorer.score(row) for row in group]
            return backend.score_shared(model, tokenizer, group, metadata)[0]

        run(first)
        mx.clear_cache()
        mx.reset_peak_memory()
        values, durations = [], []
        with (output / f"shape777-{mode}.jsonl").open("x") as stream:
            for index, group in enumerate(groups.values()):
                mark = time.perf_counter()
                scored = run(group)
                durations.append(time.perf_counter() - mark)
                values.extend(scored)
                for row in scored:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if (index + 1) % 5 == 0:
                    print(f"shape/{mode}: {index + 1}/37 states; active={mx.get_active_memory() / 2**30:.2f} GiB; cached={mx.get_cache_memory() / 2**30:.2f} GiB", flush=True)
        elapsed = sum(durations)
        predictions[mode] = values
        reports.append({"mode": mode, "wall_seconds": elapsed, "decisions_per_second": len(values) / elapsed,
                        "state_p50_seconds": statistics.median(durations), "state_seconds": durations,
                        "peak_mlx_bytes": mx.get_peak_memory()})
        print(json.dumps(reports[-1], default=str), flush=True)
    result = {"runs": reports, "vs_fresh": {mode: compare(predictions["direct"], predictions[mode])
                                            for mode in ("serial", "shared")}}
    write(output / "shape777.json", result)
    return reports


def generate(model, tokenizer, rows, max_tokens=128):
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    mx.synchronize()
    started = time.perf_counter()
    prompt = tokenizer.apply_chat_template(compact_messages(rows[0]["state"], rows), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens, events = [], []
    generator = generate_step(mx.array(ids), model, max_tokens=max_tokens)
    try:
        for token, _ in generator:
            tokens.append(token)
            events.append({"token_id": token, "seconds": time.perf_counter() - started})
            if token in tokenizer.eos_token_ids:
                break
    finally:
        generator.close()
    mx.synchronize()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    parsed = None
    try:
        candidate = json.loads(text)
        if isinstance(candidate, list) and len(candidate) == len(rows) and all(
            isinstance(value, str) and value in {"yes", "no"} for value in candidate
        ):
            parsed = candidate
    except json.JSONDecodeError:
        pass
    return {"total_seconds": time.perf_counter() - started, "input_tokens": len(ids),
            "output_tokens": len(tokens), "output_text": text, "choices": parsed,
            "valid_complete_array": parsed is not None, "timeline": events,
            "time_to_first_token_seconds": events[0]["seconds"] if events else None,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def generation(model, tokenizer, metadata, output):
    import mlx.core as mx

    rows = read("benchmarks/data/shape777.jsonl")[:21]
    backend.score_shared(model, tokenizer, rows, metadata)
    generate(model, tokenizer, rows[:1], 16)
    direct_runs, generation_runs = [], []
    for _ in range(3):
        mx.reset_peak_memory()
        values, timing = backend.score_shared(model, tokenizer, rows, metadata)
        direct_runs.append({**timing, "peak_mlx_bytes": mx.get_peak_memory(), "outputs": values})
        mx.reset_peak_memory()
        generated = generate(model, tokenizer, rows)
        generated["peak_mlx_bytes"] = mx.get_peak_memory()
        generated["agreement_with_direct"] = (sum(choice(row) == value for row, value in zip(values, generated["choices"])) / len(rows)
                                               if generated["choices"] is not None else None)
        generation_runs.append(generated)
    direct_median = statistics.median(run["total_seconds"] for run in direct_runs)
    generation_median = statistics.median(run["total_seconds"] for run in generation_runs)
    result = {"direct_runs": direct_runs, "generation_runs": generation_runs,
              "direct_median_seconds": direct_median, "generation_median_seconds": generation_median,
              "generation_over_direct_ratio": generation_median / direct_median,
              "all_generated_arrays_valid": all(run["valid_complete_array"] for run in generation_runs)}
    write(output / "generation.json", result)
    return {key: value for key, value in result.items() if not key.endswith("runs")}


def diagnostic(model, tokenizer, metadata, output):
    """Freeze a small pilot before running full quality/systems populations."""
    import mlx.core as mx
    from transformers import AutoTokenizer

    original_tokenizer = AutoTokenizer.from_pretrained(metadata["source"], revision=metadata["revision"], trust_remote_code=False)
    groups = [read("benchmarks/data/shape777.jsonl")[:4]]
    sample = read("examples/decisions.jsonl")[0]
    groups.append([dict(sample, id=f"pilot-{n}", question=sample["question"] + " " * n) for n in range(3)])
    report = []
    for rows in groups:
        for row in rows:
            if encode_prompt(tokenizer, row, 4096) != encode_prompt(original_tokenizer, row, 4096):
                raise AssertionError("MLX and original tokenizers differ")
        fresh = [backend.score(model, tokenizer, row, metadata) for row in rows]
        serial = backend.SerialPrefixScorer(model, tokenizer, metadata)
        cached = [serial.score(row) for row in rows]
        shared, _ = backend.score_shared(model, tokenizer, rows, metadata)
        reversed_shared, _ = backend.score_shared(model, tokenizer, rows[::-1], metadata)
        report.append({"fresh": fresh, "serial": cached, "shared": shared,
                       "serial_vs_fresh": compare(fresh, cached), "shared_vs_fresh": compare(fresh, shared),
                       "reordered_vs_shared": compare(shared, reversed_shared)})
    mx.synchronize()
    write(output / "diagnostic.json", report)
    return [{key: value for key, value in part.items() if "_vs_" in key} for part in report]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--bits", type=int, choices=(4, 8))
    parser.add_argument("--mlx-cache-limit-mib", type=int, default=backend.DEFAULT_CACHE_LIMIT_MIB,
                        help="Inactive allocation cache in MiB (default: 256; 0 disables caching)")
    parser.add_argument("--suite", choices=("diagnostic", "quality", "shape", "generation", "quantization", "all"), default="all")
    parser.add_argument("--reference-run", type=Path, help="Completed unquantized MLX quality run for quantization comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mlx_cache_limit_mib < 0:
        parser.error("--mlx-cache-limit-mib must be nonnegative")
    if args.suite == "quantization" and args.bits is None:
        parser.error("The quantization suite requires --bits 4 or --bits 8")
    reference_metadata = None
    if args.suite == "quantization" and args.reference_run is None:
        parser.error("The quantization suite requires --reference-run with unquantized MLX evidence")
    if args.reference_run is not None:
        reference_model = json.loads((args.reference_run / "manifest.json").read_text())["model"]
        if reference_model["quantization"] or reference_model["source"] != args.model or reference_model["revision"] != args.revision:
            parser.error("Reference must use the same unquantized source model and revision")
        reference_metadata = {"path": str(args.reference_run), "sha256": {
            name: hashlib.sha256(read_bytes(args.reference_run / name)).hexdigest()
            for name in ("manifest.json", "authored144.jsonl", "perturbations108.jsonl")}}
    args.output.mkdir(parents=True, exist_ok=False)
    model, tokenizer, metadata = backend.load_model(
        args.model, args.revision, args.bits, cache_limit_mib=args.mlx_cache_limit_mib)
    import mlx.core as mx

    manifest = {"model": metadata, "hardware": mx.device_info(), "macos": platform.mac_ver()[0],
                "python": platform.python_version(), "suite": args.suite, "reference_run": reference_metadata,
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
                    Path("benchmarks/mlx_benchmark.py"), Path("benchmarks/mlx_evidence.py"),
                    Path("benchmarks/evaluate.py"),
                    Path("benchmarks/decision_vs_generation.py"),
                    *sorted(Path("src/fastjev").rglob("*.py")),
                ]},
                "timing_scope": "Warm GPU execution including prompt preparation and CPU readout; excludes model loading and result writes. MLX operations are evaluated and synchronized.",
                "fixture_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in Path("benchmarks/data").glob("*.jsonl")}}
    write(args.output / "manifest.json", manifest)
    summaries = {}
    suites = {"all": ("diagnostic", "quality", "shape", "generation"),
              "quantization": ("diagnostic", "quality", "generation")}.get(args.suite, (args.suite,))
    for name, run in (("diagnostic", diagnostic), ("quality", quality), ("shape", shape), ("generation", generation)):
        if name in suites:
            summaries[name] = (run(model, tokenizer, metadata, args.output, args.reference_run) if name == "quality"
                               else run(model, tokenizer, metadata, args.output))
            print(json.dumps({name: summaries[name]}), flush=True)
    write(args.output / "summary.json", summaries)


if __name__ == "__main__":
    main()
