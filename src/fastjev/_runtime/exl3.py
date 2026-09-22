"""EXL3 option readout through ExLlamaV3."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import re
import time

from .core import LETTERS, digest, direct_messages, softmax
from .direct import PROMPT_VERSION


def _resolve_model(source: str, revision: str) -> tuple[Path, str]:
    if not isinstance(source, str) or not source:
        raise ValueError("model must be a nonempty local directory or Hugging Face repo ID")
    candidate = Path(source).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError("local EXL3 models must be directories")
        if not isinstance(revision, str) or not revision:
            raise ValueError("Local models require an explicit manifest/revision string")
        return candidate.resolve(), str(candidate.resolve())
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Hugging Face download support is missing; install 'fastjev[exl3]'") from error
    try:
        path = snapshot_download(repo_id=source, revision=revision)
    except Exception as error:
        raise RuntimeError(
            f"Could not download EXL3 model {source!r} at {revision!r}: {error}"
        ) from error
    return Path(path).resolve(), source


def _runtime_version() -> str:
    try:
        from exllamav3.version import __version__

        if isinstance(__version__, str) and __version__:
            return __version__
    except Exception:
        pass
    try:
        return version("exllamav3")
    except Exception:
        return "unknown"


def _encode_ids(tokenizer, text: str) -> list[int]:
    import torch

    encoded = tokenizer.encode(text, encode_special_tokens=True)
    if isinstance(encoded, tuple):
        encoded = encoded[0]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.flatten().tolist()
    return [int(token) for token in encoded]


def _slot_ids(tokenizer, count: int, prompt: str, ids: list[int]) -> list[int]:
    import torch

    slots = []
    for letter in LETTERS[:count]:
        token_id = int(tokenizer.single_id(letter))
        decoded = tokenizer.decode(torch.tensor([token_id])).strip()
        if decoded != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        if _encode_ids(tokenizer, prompt + letter) != ids + [token_id]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
        slots.append(token_id)
    if len(slots) != len(set(slots)):
        raise ValueError("Answer-slot tokens collide")
    return slots


def _encode_prompt(tokenizer, row: dict, max_tokens: int, cache_size: int):
    messages = direct_messages(row)
    prompt = tokenizer.hf_render_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = _encode_ids(tokenizer, prompt)
    if not ids or len(ids) > max_tokens:
        raise ValueError(
            f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; "
            "no truncation allowed"
        )
    if len(ids) + 1 > cache_size:
        raise ValueError(
            f"Row {row['id']}: {len(ids)} input tokens exceed EXL3 cache budget "
            f"{cache_size - 1}; no truncation allowed"
        )
    slots = _slot_ids(tokenizer, len(row["options"]), prompt, ids)
    return ids, slots, digest(prompt)


class Runtime:
    """One resident ExLlamaV3 generator and its tokenizer."""

    def __init__(self, model, config, cache, tokenizer, generator, cache_size: int):
        self.model = model
        self.config = config
        self.cache = cache
        self.tokenizer = tokenizer
        self.generator = generator
        self.cache_size = cache_size

    def close(self) -> None:
        unload = getattr(self.model, "unload", None)
        if callable(unload):
            unload()


def load_model(
    source: str,
    revision: str,
    *,
    cache_size: int = 16384,
    gpu_split: float = 22.5,
):
    """Load one pinned local or Hugging Face EXL3 checkpoint."""
    if type(cache_size) is not int or cache_size < 2:
        raise ValueError("cache_size must be an integer of at least 2")
    if (
        isinstance(gpu_split, bool)
        or not isinstance(gpu_split, (int, float))
        or not 0 < float(gpu_split)
    ):
        raise ValueError("gpu_split must be a positive number of GiB")
    path, source_identity = _resolve_model(source, revision)
    try:
        import torch
        from exllamav3 import Generator, model_init
    except ImportError as error:
        raise RuntimeError("EXL3 backend dependencies are missing; install 'fastjev[exl3]'") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES")

    parser = argparse.ArgumentParser(add_help=False)
    model_init.add_args(parser, add_draft_model_args=False)
    args = parser.parse_args([
        "-m", str(path),
        "-gs", str(float(gpu_split)),
        "-cs", str(cache_size),
    ])
    model, config, cache, tokenizer = model_init.init(args, progress=False)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    runtime = Runtime(model, config, cache, tokenizer, generator, cache_size)
    metadata = {
        "source": source_identity,
        "revision": revision,
        "backend": "exl3",
        "format": "exl3-quantized",
        "exllamav3_version": _runtime_version(),
        "torch_version": torch.__version__,
        "cache_size": cache_size,
        "gpu_split_gib": float(gpu_split),
        "resolved_model_path": str(path),
    }
    return runtime, tokenizer, metadata


def score(runtime: Runtime, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    """Read declared option scores from the last prompt position."""
    import torch
    from exllamav3 import Job

    started = time.perf_counter()
    ids, slots, prompt_hash = _encode_prompt(
        tokenizer, row, max_tokens, runtime.cache_size
    )
    job = Job(
        input_ids=torch.tensor([ids], dtype=torch.long),
        max_new_tokens=1,
        return_logits=True,
        seed=53,
    )
    forward_start = time.perf_counter()
    runtime.generator.enqueue(job)
    response = None
    while (
        runtime.generator.num_remaining_jobs()
        or runtime.generator.num_active_jobs()
    ):
        for candidate in runtime.generator.iterate():
            if candidate.get("logits") is not None:
                response = candidate
    forward_seconds = time.perf_counter() - forward_start
    if response is None:
        raise RuntimeError("ExLlamaV3 completed without returning logits")
    logits = response["logits"]
    if getattr(logits, "ndim", None) != 3 or logits.shape[0] != 1:
        raise RuntimeError(f"ExLlamaV3 returned unexpected logits shape {tuple(logits.shape)!r}")
    vocabulary = logits[0, -1, :].float().cpu()
    selected = [float(vocabulary[token_id]) for token_id in slots]
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "answer_token_ids": slots,
        "input_tokens": len(ids),
        "output_tokens": 0,
        "forward_seconds": forward_seconds,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "native ExLlamaV3 last-position logits restricted to declared answer slots",
        "probability_status": (
            "conditional option score over quantized weights; uncalibrated as decision confidence"
        ),
    }
