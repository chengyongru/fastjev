"""Same frozen FastJev workloads, pinned models, native and generic prompt paths."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
import transformers
import thisthat.model
import thisthat.prompt
from fastjev import BackendOption, BackendRequest, TorchBackend, load_causal_model
from thisthat import TypedDecider, Question
from thisthat.prompt import build

sys.path.insert(0, str(Path.cwd() / "benchmarks"))
from evaluate import align, balanced_metric, basic, clusters, paired_comparison, summarize

UPSTREAM = "542d445efa5f68b14bfbd1f8ed25aacd8379d839"
FLOCK_REVISION = "3d927195c4f9845efe66c5715883a7a0f42b1239"
QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def metrics(gold, predictions):
    aligned = align(gold, predictions)
    summary = summarize(aligned)
    summary["mean_family_balanced_accuracy"] = balanced_metric(aligned)
    summary["by_family"] = {key: basic(rows) for key, rows in clusters(aligned, "family").items()}
    summary["ece"] = sum(b["n"] * abs(b["mean_confidence"] - b["accuracy"])
                         for b in summary["reliability_bins"]) / len(aligned)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flock", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be create-only")
    assert torch.cuda.device_count() == 1
    model_path = Path(args.flock) / "model.safetensors"
    model_hash = hashlib.sha256()
    with model_path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            model_hash.update(block)
    if model_hash.hexdigest() != "11bab4bbbce0214dcb4d70a88e74b3e4bde6fe8a95f239e3d0c355946e0000e0":
        parser.error("--flock weights do not match the pinned this-that checkpoint")
    datasets = {name: [json.loads(line) for line in Path(f"benchmarks/data/{name}.jsonl").read_text().splitlines()]
                for name in ("authored144", "perturbations108")}
    report = {
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "python": platform.python_version(), "gpu": torch.cuda.get_device_name()},
        "thisthat_source_revision": UPSTREAM,
        "thisthat_weights_sha256": model_hash.hexdigest(),
        "source_hash_encoding": "UTF-8 with LF line endings",
        "script_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
                          for name, path in {
            "thisthat/model.py": Path(thisthat.model.__file__),
            "thisthat/prompt.py": Path(thisthat.prompt.__file__),
            "fastjev/client.py": Path("src/fastjev/client.py"),
            "fastjev/backends/torch.py": Path("src/fastjev/backends/torch.py"),
            "fastjev/_runtime/direct.py": Path("src/fastjev/_runtime/direct.py"),
        }.items()},
        "datasets": {name: {"rows": len(rows), "sha256": hashlib.sha256(
            Path(f"benchmarks/data/{name}.jsonl").read_bytes()).hexdigest()} for name, rows in datasets.items()},
        "modes": {},
    }
    for name, source, revision in [("qwen4b", "Qwen/Qwen3.5-4B", QWEN_REVISION),
                                    ("thisthat", args.flock, FLOCK_REVISION)]:
        started = time.perf_counter()
        model, tokenizer, metadata = load_causal_model(source, revision, "cuda", "bfloat16")
        load_seconds = time.perf_counter() - started
        for mode in (["direct", "batch8"] if name == "qwen4b" else ["direct", "batch8", "native"]):
            mode_key = f"{name}_{mode}"
            backend = TorchBackend(model, tokenizer, metadata, batch_size=8 if mode == "batch8" else 1,
                                   sort_by_length=mode == "batch8")
            native = TypedDecider(model, tokenizer) if mode == "native" else None
            def score_rows(rows):
                if native is None:
                    requests = [BackendRequest(row["id"], row["state"], row["question"], tuple(
                        BackendOption(o["id"], o["description"]) for o in row["options"])) for row in rows]
                    return [asdict(value) for value in backend.score(requests)]
                outputs = []
                for row in rows:
                    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
                    questions = [Question(row["question"], [o["description"] for o in row["options"]])]
                    assert len(tokenizer.encode("Context:\n" + state, add_special_tokens=False)) <= 4099
                    prompt = build(tokenizer, state, questions, max_state_tokens=4096)
                    assert len(prompt["ids"]) <= 4096
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    result = native.decide(state, questions[0], max_state_tokens=4096, temperature=1.0)
                    torch.cuda.synchronize()
                    outputs.append({"id": row["id"], "option_ids": [o["id"] for o in row["options"]],
                                    "probabilities": list(result.probabilities), "input_tokens": len(prompt["ids"]),
                                    "output_tokens": 0, "total_seconds": time.perf_counter()-t0,
                                    "prompt_version": "thisthat-state-first@" + UPSTREAM,
                                    "probability_status": "native temperature=1; not calibrated on this workload"})
                return outputs
            score_rows(datasets["authored144"][:8])
            torch.cuda.reset_peak_memory_stats()
            entry = {"model": metadata, "load_seconds": load_seconds, "datasets": {}}
            for dataset, rows in datasets.items():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                predictions = score_rows(rows)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                # JSON-normalize transport tuples before applying the repository metric contract.
                predictions = json.loads(json.dumps(predictions))
                entry["datasets"][dataset] = {"metrics": metrics(rows, predictions),
                    "wall_seconds": elapsed, "decisions_per_second": len(rows) / elapsed,
                    "predictions": predictions}
                print(mode_key, dataset, json.dumps({"seconds": elapsed,
                    "metrics": entry["datasets"][dataset]["metrics"]}), flush=True)
            entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["modes"][mode_key] = entry
            backend.close()
            del native, backend
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    report["comparisons"] = {}
    for dataset, rows in datasets.items():
        base = report["modes"]["qwen4b_direct"]["datasets"][dataset]["predictions"]
        report["comparisons"][dataset] = {
            key: paired_comparison(align(rows, value["datasets"][dataset]["predictions"]), align(rows, base))
            for key, value in report["modes"].items() if key != "qwen4b_direct"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
