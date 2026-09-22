"""Create-only JSONL command line scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._runtime.core import load_causal_model, validate_row
from ._runtime.direct import score as direct_score
from ._runtime.reranker import score as reranker_score
from ._runtime.serial import SerialPrefixScorer
from ._runtime.shared import score_shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "serial", "shared", "reranker"), required=True)
    parser.add_argument("--backend", choices=("torch", "mlx", "llama-cpp"), default="torch")
    parser.add_argument("--mlx-bits", type=int, choices=(4, 8), help="Quantize MLX weights in memory; default preserves source precision")
    parser.add_argument("--mlx-cache-limit-mib", type=int,
                        help="MLX inactive allocation cache in MiB (default: 256; 0 disables caching)")
    parser.add_argument("--llama-cpp-n-gpu-layers", type=int, default=-1,
                        help="llama.cpp layers to offload; -1 means all layers")
    parser.add_argument("--llama-cpp-n-batch", type=int, default=512,
                        help="llama.cpp prompt batch size")
    parser.add_argument("--llama-cpp-filename",
                        help="Exact GGUF filename when --model is a Hugging Face repo ID")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    if args.llama_cpp_n_gpu_layers < -1:
        parser.error("--llama-cpp-n-gpu-layers must be -1 or nonnegative")
    if args.llama_cpp_n_batch < 1:
        parser.error("--llama-cpp-n-batch must be positive")
    if args.llama_cpp_filename is not None and args.backend != "llama-cpp":
        parser.error("--llama-cpp-filename requires --backend llama-cpp")
    if args.mlx_bits and args.backend != "mlx":
        parser.error("--mlx-bits requires --backend mlx")
    if args.mlx_cache_limit_mib is not None:
        if args.backend != "mlx":
            parser.error("--mlx-cache-limit-mib requires --backend mlx")
        if args.mlx_cache_limit_mib < 0:
            parser.error("--mlx-cache-limit-mib must be nonnegative")
    if args.backend == "mlx" and args.mode == "reranker":
        parser.error("MLX supports direct, serial, and shared modes; reranker requires torch")
    if args.backend == "llama-cpp" and args.mode != "direct":
        parser.error("llama-cpp supports direct mode only")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    direct, serial, shared = direct_score, SerialPrefixScorer, score_shared
    if args.backend == "mlx":
        from ._runtime import mlx

        cache_limit_mib = (mlx.DEFAULT_CACHE_LIMIT_MIB if args.mlx_cache_limit_mib is None
                           else args.mlx_cache_limit_mib)
        model, tokenizer, metadata = mlx.load_model(
            args.model, args.revision, args.mlx_bits, cache_limit_mib=cache_limit_mib)
        direct, serial, shared = mlx.score, mlx.SerialPrefixScorer, mlx.score_shared
    elif args.backend == "llama-cpp":
        from ._runtime import llama_cpp

        model, tokenizer, metadata = llama_cpp.load_model(
            args.model,
            args.revision,
            filename=args.llama_cpp_filename,
            max_input_tokens=args.max_tokens,
            n_batch=args.llama_cpp_n_batch,
            n_gpu_layers=args.llama_cpp_n_gpu_layers,
        )
        direct = llama_cpp.score
    else:
        model, tokenizer, metadata = load_causal_model(args.model, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        if args.mode == "shared":
            results, timing = shared(model, tokenizer, rows, metadata, args.max_tokens)
            for result in results:
                destination.write(json.dumps({**result, "shared_timing": timing}, allow_nan=False) + "\n")
        elif args.mode == "serial":
            scorer = serial(model, tokenizer, metadata, args.max_tokens)
            for row in rows:
                destination.write(json.dumps(scorer.score(row), allow_nan=False) + "\n")
                destination.flush()
        else:
            scorer = direct if args.mode == "direct" else reranker_score
            for row in rows:
                destination.write(json.dumps(scorer(model, tokenizer, row, metadata, args.max_tokens), allow_nan=False) + "\n")
                destination.flush()


if __name__ == "__main__":
    main()
