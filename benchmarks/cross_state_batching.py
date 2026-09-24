"""Compare sequential, padded, and length-grouped Torch calls through the public SDK."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from fastjev import Boolean, Choice, FastJev, Level, Option, Score, TorchBackend, load_causal_model


QUESTIONS = {
    "department": Choice("Which team should handle the customer's request?", [
        Option("billing", "Invoices, duplicate charges, payments, and refunds."),
        Option("access", "Login failures, passwords, and account access."),
        Option("technical", "Application bugs, outages, and errors."),
    ]),
    "refund": Boolean("Does the customer explicitly request a refund?"),
    "urgency": Score("How urgent is the request?", [
        Level(0, "Routine question; no blocked work or deadline."),
        Level(1, "An inconvenience, but work can continue."),
        Level(2, "Work is blocked or an immediate deadline is stated."),
    ]),
}
MESSAGES = [
    "I was charged twice. Please refund the duplicate payment today.",
    "I cannot log in and my entire team is blocked. Please restore account access now.",
    "The export button occasionally shows an error; retrying works. There is no deadline.",
    "Please explain where I can find last month's invoice. This is not urgent.",
    "我被重复扣款了，请退还多扣的钱。",
    "登录失败，全组都无法工作，请立即恢复访问。",
]


def run(engine, states, *, serial):
    torch.cuda.synchronize()
    started = time.perf_counter()
    decisions = ([engine.decide_many(state, QUESTIONS) for state in states] if serial
                 else engine.decide_batch(states, QUESTIONS))
    torch.cuda.synchronize()
    return {
        "wall_seconds": time.perf_counter() - started,
        "decisions": [{key: asdict(value) for key, value in item.items()} for item in decisions],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new path")
    if min(args.states, args.repeats) < 1 or args.batch_size < 2:
        parser.error("positive states/repeats and batch-size >= 2 are required")
    if torch.cuda.device_count() != 1:
        parser.error("expose exactly one CUDA GPU")
    states = [{
        "message": MESSAGES[i % len(MESSAGES)],
        "unrelated_archive": "The office inventory lists a blue mug and a wooden shelf. " * (
            (i * 17) % 61),
    } for i in range(args.states)]
    started = time.perf_counter()
    model, tokenizer, metadata = load_causal_model(args.model, args.revision, "cuda", "bfloat16")
    report = {
        "model": metadata,
        "load_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "gpu": torch.cuda.get_device_name(0),
                        "causal_conv1d_installed": importlib.util.find_spec("causal_conv1d") is not None,
                        "flash_linear_attention_installed": importlib.util.find_spec("fla") is not None},
        "source_hash_encoding": "UTF-8 with LF line endings",
        "source_sha256": {str(path): hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest() for path in [
            Path("src/fastjev/client.py"), Path("src/fastjev/backends/torch.py"),
            Path("src/fastjev/_runtime/direct.py"), Path(__file__).relative_to(Path.cwd()),
        ]},
        "states": states,
        "questions": {key: asdict(value) for key, value in QUESTIONS.items()},
        "repeats": args.repeats,
        "warmups_per_mode": 1,
        "modes": {},
    }
    baseline = None
    for name, size, sort in [("sequential", 1, False), ("batched", args.batch_size, False),
                             ("length_grouped", args.batch_size, True)]:
        with FastJev(TorchBackend(model, tokenizer, metadata, batch_size=size,
                                 sort_by_length=sort)) as engine:
            run(engine, states, serial=size == 1)
            torch.cuda.reset_peak_memory_stats()
            runs = [run(engine, states, serial=size == 1) for _ in range(args.repeats)]
            peak = torch.cuda.max_memory_allocated()
        if baseline is None:
            baseline = runs[0]["decisions"]
        disagreements = 0
        drifts = []
        for measurement in runs:
            for expected, actual in zip(baseline, measurement["decisions"]):
                for key in QUESTIONS:
                    disagreements += expected[key]["selected"] != actual[key]["selected"]
                    drifts.extend(abs(value - actual[key]["probabilities"][option])
                                  for option, value in expected[key]["probabilities"].items())
        median = statistics.median(item["wall_seconds"] for item in runs)
        summary = {
            "batch_size": size, "sort_by_length": sort,
            "median_seconds": median,
            "decisions_per_second": len(states) * len(QUESTIONS) / median,
            "peak_allocated_bytes": peak,
            "argmax_disagreements_vs_first_sequential": disagreements,
            "max_probability_drift": max(drifts),
            "mean_probability_drift": statistics.mean(drifts),
        }
        report["modes"][name] = {**summary, "runs": runs}
        print(name, json.dumps(summary), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
