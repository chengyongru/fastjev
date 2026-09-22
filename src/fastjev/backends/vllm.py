"""vLLM batched option-scoring backend for the public fastjev SDK."""

from __future__ import annotations

import math
from pathlib import Path
import re
import time
from typing import Any, Sequence

from ..runtime.direct import PROMPT_VERSION, encode_prompt

from ..errors import (
    BackendExecutionError,
    BackendProtocolError,
    InputTooLongError,
    ModelLoadError,
    ValidationError,
)
from .base import (
    BackendCapabilities,
    BackendInfo,
    BackendRequest,
    BackendResult,
)


class VLLMBackend:
    """Batch categorical requests through one resident offline vLLM engine."""

    def __init__(self, engine: Any, tokenizer: Any, metadata: dict, *, max_input_tokens: int = 4096):
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValidationError("max_input_tokens must be a positive integer")
        source = metadata.get("source")
        revision = metadata.get("revision")
        if not isinstance(source, str) or not source or not isinstance(revision, str) or not revision:
            raise ValidationError("vLLM metadata must contain nonempty source and revision strings")
        self._engine = engine
        self._tokenizer = tokenizer
        self._metadata = dict(metadata)
        self._max_input_tokens = max_input_tokens
        self._closed = False
        self._info = BackendInfo(name="vllm", model=source, revision=revision)
        self._capabilities = BackendCapabilities(min_options=2, max_options=16)

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        revision: str,
        *,
        max_input_tokens: int = 4096,
        **engine_kwargs: Any,
    ) -> "VLLMBackend":
        """Load a pinned checkpoint through the optional vLLM runtime."""
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValidationError("max_input_tokens must be a positive integer")
        if not isinstance(model, str) or not model:
            raise ValidationError("model must be a nonempty string")
        local = Path(model).exists()
        if not isinstance(revision, str) or not revision:
            raise ValidationError("revision must be a nonempty string")
        if not local and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValidationError("Remote models require a pinned 40-character commit revision")
        reserved = {"model", "revision", "tokenizer_revision", "trust_remote_code", "max_model_len"}
        overlap = reserved.intersection(engine_kwargs)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValidationError(f"vLLM engine options are managed by fastjev: {names}")

        try:
            import vllm

            runtime_revision = None if local else revision
            engine = vllm.LLM(
                model=model,
                revision=runtime_revision,
                tokenizer_revision=runtime_revision,
                trust_remote_code=False,
                max_model_len=max_input_tokens + 1,
                **engine_kwargs,
            )
            tokenizer = engine.get_tokenizer()
        except ImportError as error:
            raise ModelLoadError(
                "vLLM backend dependencies are missing; install 'fastjev[vllm]'"
            ) from error
        except Exception as error:
            raise ModelLoadError(str(error)) from error

        metadata = {
            "source": model,
            "revision": revision,
            "vllm_version": getattr(vllm, "__version__", "unknown"),
        }
        return cls(engine, tokenizer, metadata, max_input_tokens=max_input_tokens)

    @property
    def info(self) -> BackendInfo:
        return self._info

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def score(self, requests: Sequence[BackendRequest]) -> list[BackendResult]:
        if self._closed:
            raise RuntimeError("VLLMBackend is closed")
        if not requests:
            return []

        try:
            from vllm import SamplingParams
        except ImportError as error:
            raise ModelLoadError(
                "vLLM backend dependencies are missing; install 'fastjev[vllm]'"
            ) from error

        prompts = []
        params = []
        prompt_ids = []
        slot_ids = []
        for request in requests:
            row = {
                "id": request.id,
                "state": request.state,
                "question": request.instruction,
                "options": [
                    {"id": option.id, "description": option.description}
                    for option in request.options
                ],
            }
            try:
                ids, slots, _prompt_hash = encode_prompt(
                    self._tokenizer,
                    row,
                    self._max_input_tokens,
                )
            except ValueError as error:
                if "input tokens exceed limit" in str(error):
                    raise InputTooLongError(str(error)) from error
                raise ValidationError(str(error)) from error
            prompt_ids.append(ids)
            slot_ids.append(slots)
            prompts.append({"prompt_token_ids": ids})
            params.append(SamplingParams(
                max_tokens=1,
                temperature=1.0,
                allowed_token_ids=slots,
                logprob_token_ids=slots,
                detokenize=False,
                seed=0,
            ))

        started = time.perf_counter()
        try:
            outputs = list(self._engine.generate(prompts, params, use_tqdm=False))
        except Exception as error:
            raise BackendExecutionError(str(error)) from error
        elapsed = time.perf_counter() - started
        if len(outputs) != len(requests):
            raise BackendProtocolError(
                "vLLM returned a different number of results than requested"
            )

        results = []
        for request, ids, slots, output in zip(requests, prompt_ids, slot_ids, outputs):
            completions = getattr(output, "outputs", None)
            if not isinstance(completions, list) or len(completions) != 1:
                raise BackendProtocolError(
                    f"vLLM result for {request.id!r} must contain one completion"
                )
            completion = completions[0]
            logprobs = getattr(completion, "logprobs", None)
            if logprobs is None or len(logprobs) != 1:
                raise BackendProtocolError(
                    f"vLLM result for {request.id!r} must contain one logprob position"
                )
            position = logprobs[0]
            try:
                probabilities = tuple(math.exp(float(position[token].logprob)) for token in slots)
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise BackendProtocolError(
                    f"vLLM result for {request.id!r} is missing requested option logprobs"
                ) from error
            results.append(BackendResult(
                id=request.id,
                option_ids=tuple(option.id for option in request.options),
                probabilities=probabilities,
                input_tokens=len(ids),
                output_tokens=1,
                total_seconds=elapsed,
                prompt_version=PROMPT_VERSION,
                probability_status=(
                    "vLLM allowed-token next-token scores; uncalibrated as decision confidence"
                ),
            ))
        return results

    def close(self) -> None:
        self._closed = True
        self._engine = None
        self._tokenizer = None
