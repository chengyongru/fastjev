"""Run one resident fastjev model behind a System One-compatible HTTP API."""

from __future__ import annotations

import argparse
from datetime import date
import os

from .compat.wire import SystemOneService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "mlx", "llama-cpp"), default="torch")
    parser.add_argument("--model", required=True, help="Hugging Face model ID, local checkpoint, or GGUF file")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--served-model", required=True, help="Model ID accepted by the HTTP API")
    parser.add_argument("--served-model-description", required=True)
    parser.add_argument("--served-model-release-date", required=True, help="ISO date returned by GET /v1/models")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key-env", default="FASTJEV_API_KEY")
    parser.add_argument("--allow-unauthenticated", action="store_true")
    parser.add_argument("--mlx-bits", type=int, choices=(4, 8))
    parser.add_argument("--mlx-cache-limit-mib", type=int)
    parser.add_argument("--llama-cpp-n-gpu-layers", type=int, default=-1)
    parser.add_argument("--llama-cpp-n-batch", type=int, default=512)
    parser.add_argument("--llama-cpp-filename",
                        help="Exact GGUF filename when --model is a Hugging Face repo ID")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args) -> str | None:
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    n_gpu_layers = getattr(args, "llama_cpp_n_gpu_layers", -1)
    n_batch = getattr(args, "llama_cpp_n_batch", 512)
    if n_gpu_layers < -1:
        parser.error("--llama-cpp-n-gpu-layers must be -1 or nonnegative")
    if n_batch < 1:
        parser.error("--llama-cpp-n-batch must be positive")
    if getattr(args, "llama_cpp_filename", None) is not None and args.backend != "llama-cpp":
        parser.error("--llama-cpp-filename requires --backend llama-cpp")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.mlx_bits and args.backend != "mlx":
        parser.error("--mlx-bits requires --backend mlx")
    if args.mlx_cache_limit_mib is not None:
        if args.backend != "mlx":
            parser.error("--mlx-cache-limit-mib requires --backend mlx")
        if args.mlx_cache_limit_mib < 0:
            parser.error("--mlx-cache-limit-mib must be nonnegative")
    if args.served_model.lower().startswith("jev"):
        parser.error("--served-model must identify fastjev and must not impersonate a Jev model or alias")
    if not args.served_model_description:
        parser.error("--served-model-description must not be empty")
    try:
        date.fromisoformat(args.served_model_release_date)
    except ValueError:
        parser.error("--served-model-release-date must be an ISO date such as 2026-09-18")
    api_key = os.environ.get(args.api_key_env)
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if not api_key and args.host not in local_hosts and not args.allow_unauthenticated:
        parser.error(
            f"{args.api_key_env} is unset; use a bearer token or explicitly pass --allow-unauthenticated"
        )
    return api_key or None


def _scorer(args):
    if args.backend == "mlx":
        from .runtime import mlx

        cache_limit = (mlx.DEFAULT_CACHE_LIMIT_MIB if args.mlx_cache_limit_mib is None
                       else args.mlx_cache_limit_mib)
        model, tokenizer, metadata = mlx.load_model(
            args.model, args.revision, args.mlx_bits, cache_limit_mib=cache_limit
        )
        direct = mlx.score
    elif args.backend == "llama-cpp":
        from .runtime import llama_cpp

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
        from .runtime.core import load_causal_model
        from .runtime.direct import score as direct

        model, tokenizer, metadata = load_causal_model(args.model, args.revision)

    def score_rows(rows):
        return [direct(model, tokenizer, row, metadata, args.max_tokens) for row in rows]

    return score_rows


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    api_key = _validate_args(parser, args)
    try:
        from .http import create_app
        import uvicorn
    except ImportError as error:
        parser.error(f"HTTP dependencies are missing; install '.[api]' ({error})")

    score_rows = _scorer(args)
    service = SystemOneService(
        score_rows=score_rows,
        served_model=args.served_model,
        description=args.served_model_description,
        release_date=args.served_model_release_date,
    )
    app = create_app(service, api_key=api_key)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
