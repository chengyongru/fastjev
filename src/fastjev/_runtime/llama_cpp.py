"""GGUF option readout through the llama.cpp Python bindings."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
import re
import time

from .core import LETTERS, digest, direct_messages, softmax
from .direct import PROMPT_VERSION


def _resolve_local_model(source: str) -> Path:
    path = Path(source).expanduser()
    if path.is_dir():
        candidates = sorted(path.glob("*.gguf"))
        if len(candidates) != 1:
            raise ValueError(
                "model directory must contain exactly one .gguf file; "
                f"found {len(candidates)}"
            )
        path = candidates[0]
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise ValueError("model must point to an existing .gguf file")
    return path.resolve()


def _resolve_model(
    source: str,
    revision: str,
    filename: str | None,
) -> tuple[Path, str, str | None]:
    if not isinstance(source, str) or not source:
        raise ValueError("model must be a nonempty local path or Hugging Face repo ID")
    candidate = Path(source).expanduser()
    if candidate.exists():
        if filename is not None:
            raise ValueError("filename is only valid for a Hugging Face GGUF repository")
        path = _resolve_local_model(source)
        return path, str(path), None

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "Remote GGUF models require a pinned 40-character commit revision"
        )
    if not isinstance(filename, str) or not filename or Path(filename).suffix.lower() != ".gguf":
        raise ValueError(
            "Remote GGUF models require the exact .gguf filename; "
            "fastjev does not choose a quantization automatically"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face download support is missing; install 'fastjev[llama-cpp]'"
        ) from error
    try:
        downloaded = hf_hub_download(
            repo_id=source,
            filename=filename,
            revision=revision,
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not download {filename!r} from Hugging Face repo {source!r}: {error}"
        ) from error
    return _resolve_local_model(downloaded), source, filename


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _runtime_version(module) -> str:
    value = getattr(module, "__version__", None)
    if isinstance(value, str) and value:
        return value
    try:
        return version("llama-cpp-python")
    except Exception:
        return "unknown"


def _decode_token(model, token_id: int) -> str:
    if token_id < 0:
        return ""
    return model.detokenize([token_id], special=False).decode("utf-8", errors="ignore")


def _chat_formatter(model):
    import llama_cpp.llama_chat_format as chat_format

    metadata = getattr(model, "metadata", {})
    template = metadata.get("tokenizer.chat_template")
    if template is None:
        template = metadata.get("tokenizer.chat_template.default")
    if isinstance(template, dict):
        template = template.get("default") or next(iter(template.values()), None)
    if not isinstance(template, str) or not template:
        raise ValueError(
            "GGUF model does not contain a tokenizer chat template; "
            "direct scoring cannot safely infer its prompt format"
        )
    return chat_format.Jinja2ChatFormatter(
        template=template,
        eos_token=_decode_token(model, model.token_eos()),
        bos_token=_decode_token(model, model.token_bos()),
        add_generation_prompt=True,
        stop_token_ids=[model.token_eos()],
    )


def _encode_prompt(model, row: dict, max_tokens: int) -> tuple[list[int], list[int], str]:
    formatter = _chat_formatter(model)
    rendered = formatter(messages=direct_messages(row), enable_thinking=False)
    prompt = rendered.prompt
    add_bos = not rendered.added_special
    ids = model.tokenize(prompt.encode("utf-8"), add_bos=add_bos, special=True)
    if not ids or len(ids) > max_tokens:
        raise ValueError(
            f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; "
            "no truncation allowed"
        )

    slots = []
    for letter in LETTERS[: len(row["options"])]:
        with_letter = model.tokenize(
            (prompt + letter).encode("utf-8"),
            add_bos=add_bos,
            special=True,
        )
        if len(with_letter) != len(ids) + 1 or with_letter[:-1] != ids:
            raise ValueError(
                f"Answer boundary changes tokenization for slot {letter!r}"
            )
        token_id = with_letter[-1]
        if _decode_token(model, token_id) != letter:
            raise ValueError(
                f"Answer slot {letter!r} is not one exact round-trip token"
            )
        slots.append(token_id)
    if len(slots) != len(set(slots)):
        raise ValueError("Answer-slot tokens collide")
    return ids, slots, digest(prompt)


def load_model(
    source: str,
    revision: str,
    *,
    filename: str | None = None,
    max_input_tokens: int = 4096,
    n_ctx: int | None = None,
    n_batch: int = 512,
    n_gpu_layers: int = -1,
    **llama_kwargs,
):
    """Load one local or Hugging Face GGUF with logits retained for direct scoring."""
    if type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ValueError("max_input_tokens must be a positive integer")
    if not isinstance(revision, str) or not revision:
        raise ValueError("revision must be a nonempty provenance label or commit")
    if type(n_batch) is not int or n_batch < 1:
        raise ValueError("n_batch must be a positive integer")
    if type(n_gpu_layers) is not int or n_gpu_layers < -1:
        raise ValueError("n_gpu_layers must be -1 or a nonnegative integer")
    path, source_identity, source_filename = _resolve_model(source, revision, filename)
    context = max(max_input_tokens + 1, n_batch) if n_ctx is None else n_ctx
    if type(context) is not int or context < max_input_tokens + 1:
        raise ValueError("n_ctx must be at least max_input_tokens + 1")
    if n_batch > context:
        raise ValueError("n_batch must not exceed n_ctx")
    reserved = {"model_path", "n_ctx", "n_batch", "n_gpu_layers", "logits_all", "verbose"}
    overlap = reserved.intersection(llama_kwargs)
    if overlap:
        raise ValueError(
            "llama.cpp options are managed by fastjev: " + ", ".join(sorted(overlap))
        )

    try:
        import llama_cpp

        model = llama_cpp.Llama(
            model_path=str(path),
            n_ctx=context,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            logits_all=True,
            verbose=False,
            **llama_kwargs,
        )
    except ImportError as error:
        raise RuntimeError(
            "llama.cpp backend dependencies are missing; install 'fastjev[llama-cpp]'"
        ) from error
    metadata = {
        "source": source_identity,
        "revision": revision,
        "backend": "llama-cpp",
        "llama_cpp_version": _runtime_version(llama_cpp),
        "format": "gguf",
        "context_length": context,
        "n_batch": n_batch,
        "n_gpu_layers": n_gpu_layers,
        "resolved_model_path": str(path),
        "source_filename": source_filename,
        "source_artifact_sha256": _sha256(path),
    }
    return model, None, metadata


def score(model, _tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    started = time.perf_counter()
    ids, slots, prompt_hash = _encode_prompt(model, row, max_tokens)
    model.reset()
    forward_start = time.perf_counter()
    model.eval(ids)
    logits = model.eval_logits[-1]
    forward_seconds = time.perf_counter() - forward_start
    selected = [float(logits[token_id]) for token_id in slots]
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
        "readout": "native llama.cpp last-position logits restricted to declared answer slots",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }
