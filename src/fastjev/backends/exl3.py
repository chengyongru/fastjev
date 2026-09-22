"""EXL3 backend for the public FastJev SDK."""

from __future__ import annotations

from typing import Sequence

from .._runtime import exl3
from ..errors import BackendExecutionError, InputTooLongError, ModelLoadError, ValidationError
from .base import BackendCapabilities, BackendInfo, BackendRequest, BackendResult


class ExLlamaV3Backend:
    """Score categorical requests with one resident EXL3-quantized model."""

    def __init__(self, runtime, tokenizer, metadata: dict, *, max_input_tokens: int = 4096):
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValidationError("max_input_tokens must be a positive integer")
        source = metadata.get("source")
        revision = metadata.get("revision")
        if not isinstance(source, str) or not source or not isinstance(revision, str) or not revision:
            raise ValidationError("EXL3 metadata must contain nonempty source and revision strings")
        self._runtime = runtime
        self._tokenizer = tokenizer
        self._metadata = dict(metadata)
        self._max_input_tokens = max_input_tokens
        self._closed = False
        self._info = BackendInfo(name="exl3", model=source, revision=revision)
        self._capabilities = BackendCapabilities(min_options=2, max_options=16)

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        revision: str,
        *,
        max_input_tokens: int = 4096,
        cache_size: int = 16384,
        gpu_split: float = 22.5,
    ) -> "ExLlamaV3Backend":
        """Load a local or pinned Hugging Face EXL3 checkpoint."""
        try:
            runtime, tokenizer, metadata = exl3.load_model(
                model,
                revision,
                cache_size=cache_size,
                gpu_split=gpu_split,
            )
        except ImportError as error:
            raise ModelLoadError(
                "EXL3 backend dependencies are missing; install 'fastjev[exl3]'"
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise ModelLoadError(str(error)) from error
        return cls(runtime, tokenizer, metadata, max_input_tokens=max_input_tokens)

    @property
    def info(self) -> BackendInfo:
        return self._info

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def score(self, requests: Sequence[BackendRequest]) -> list[BackendResult]:
        if self._closed:
            raise RuntimeError("ExLlamaV3Backend is closed")
        results = []
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
                result = exl3.score(
                    self._runtime,
                    self._tokenizer,
                    row,
                    self._metadata,
                    self._max_input_tokens,
                )
            except ValueError as error:
                if "input tokens exceed" in str(error):
                    raise InputTooLongError(str(error)) from error
                raise ValidationError(str(error)) from error
            except (OSError, RuntimeError) as error:
                raise BackendExecutionError(str(error)) from error
            results.append(BackendResult(
                id=result["id"],
                option_ids=tuple(result["option_ids"]),
                probabilities=tuple(result["probabilities"]),
                input_tokens=int(result.get("input_tokens", 0)),
                output_tokens=int(result.get("output_tokens", 0)),
                total_seconds=result.get("total_seconds"),
                forward_seconds=result.get("forward_seconds"),
                prompt_version=result.get("prompt_version"),
                probability_status=result.get(
                    "probability_status", "conditional option score; uncalibrated"
                ),
            ))
        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._runtime is not None:
            close = getattr(self._runtime, "close", None)
            if callable(close):
                close()
        self._runtime = None
        self._tokenizer = None

