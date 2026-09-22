"""Measure the frozen SDK safeguard workload through Torch or EXL3."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

import fastjev
from fastjev import (
    ExLlamaV3Backend,
    FastJev,
    TemperatureCalibration,
    TorchBackend,
)

from llama_cpp_sdk import EXPECTED, QUESTIONS, STATE, decision_record, question_record


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def gpu_snapshot() -> dict | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    rows = []
    for line in output.splitlines():
        name, driver, total, used = (value.strip() for value in line.split(",", 3))
        rows.append({
            "name": name,
            "driver_version": driver,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
        })
    return {"gpus": rows}


def run_decisions(engine: FastJev) -> dict:
    started = time.perf_counter()
    decisions = engine.decide_many(STATE, QUESTIONS)
    wall_seconds = time.perf_counter() - started
    return {
        "wall_seconds": wall_seconds,
        "decisions": {key: decision_record(value) for key, value in decisions.items()},
        "matches_expected": {
            key: decisions[key].selected == expected for key, expected in EXPECTED.items()
        },
        "all_calibrated": all(value.uncertainty.calibrated for value in decisions.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "exl3"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--fastjev-commit", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--exl3-cache-size", type=int, default=16384)
    parser.add_argument("--exl3-gpu-split", type=float, default=22.5)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if args.warmups < 1 or args.repeats < 1 or args.max_input_tokens < 1:
        parser.error("warmups, repeats, and max-input-tokens must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.fastjev_commit):
        parser.error("fastjev-commit must be a full 40-character Git commit")

    before_load = gpu_snapshot()
    load_started = time.perf_counter()
    if args.backend == "torch":
        backend = TorchBackend.from_pretrained(
            args.model,
            args.revision,
            max_input_tokens=args.max_input_tokens,
            device="cuda",
            dtype=args.dtype,
        )
    else:
        backend = ExLlamaV3Backend.from_pretrained(
            args.model,
            args.revision,
            max_input_tokens=args.max_input_tokens,
            cache_size=args.exl3_cache_size,
            gpu_split=args.exl3_gpu_split,
        )
    load_seconds = time.perf_counter() - load_started
    after_load = gpu_snapshot()
    calibration = None
    if args.temperature is not None:
        calibration = TemperatureCalibration(
            temperature=args.temperature,
            workload="shell-safeguard-v1",
            backend=backend.info.name,
            model=backend.info.model,
            revision=backend.info.revision,
            prompt_version="direct-options-v1",
        )
    try:
        with FastJev(backend, calibration=calibration) as engine:
            warmups = [run_decisions(engine) for _ in range(args.warmups)]
            measurements = [run_decisions(engine) for _ in range(args.repeats)]
            after_measurement = gpu_snapshot()
    finally:
        backend.close()

    times = [row["wall_seconds"] for row in measurements]
    median_seconds = statistics.median(times)
    all_results = warmups + measurements
    result = {
        "schema_version": 1,
        "scope": (
            "Frozen three-question public SDK shell-safeguard smoke and warm latency; "
            "no command was executed."
        ),
        "fastjev": {
            "commit": args.fastjev_commit,
            "source_version": fastjev.__version__,
            "installed_distribution_version": package_version("fastjev"),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "backend": args.backend,
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "exllamav3": package_version("exllamav3"),
            "dtype": args.dtype if args.backend == "torch" else None,
            "exl3_cache_size": args.exl3_cache_size if args.backend == "exl3" else None,
            "exl3_gpu_split_gib": args.exl3_gpu_split if args.backend == "exl3" else None,
        },
        "model": {"source": args.model, "revision": args.revision},
        "gpu": {
            "before_load": before_load,
            "after_load": after_load,
            "after_measurement": after_measurement,
        },
        "calibration": (
            None
            if calibration is None
            else {
                "method": "temperature_scaling",
                "temperature": calibration.temperature,
                "workload": calibration.workload,
                "identity_bound": True,
            }
        ),
        "workload": {
            "state": STATE,
            "questions": {key: question_record(value) for key, value in QUESTIONS.items()},
            "expected": EXPECTED,
            "decision_count": len(QUESTIONS),
        },
        "load_seconds": load_seconds,
        "warmups": warmups,
        "measurements": measurements,
        "summary": {
            "warmup_count": args.warmups,
            "repeat_count": args.repeats,
            "median_batch_seconds": median_seconds,
            "median_batch_milliseconds": median_seconds * 1000,
            "decisions_per_second_at_median": len(QUESTIONS) / median_seconds,
            "all_expected_selections": all(
                all(row["matches_expected"].values()) for row in all_results
            ),
            "all_output_tokens_zero": all(
                decision["output_tokens"] == 0
                for row in all_results
                for decision in row["decisions"].values()
            ),
            "calibration_applied": all(
                row["all_calibrated"] for row in all_results
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()

