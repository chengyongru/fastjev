"""llama.cpp GGUF direct-logit backend for the public fastjev SDK."""

from __future__ import annotations

from typing import Any, Sequence

from semif_phase1 import llama_cpp_backend

from ..errors import BackendExecutionError, InputTooLongError, ModelLoadError, ValidationError
from .base import BackendCapabilities, BackendInfo, BackendRequest, BackendResult


class LlamaCppBackend:
    """Score categorical requests with one resident GGUF model."""

    def __init__(self, model: Any, tokenizer: Any, metadata: dict, *, max_input_tokens: int = 4096):
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValidationError("max_input_tokens must be a positive integer")
        source = metadata.get("source")
        revision = metadata.get("revision")
        if not isinstance(source, str) or not source or not isinstance(revision, str) or not revision:
            raise ValidationError("llama.cpp metadata must contain nonempty source and revision strings")
        self._model = model
        self._tokenizer = tokenizer
        self._metadata = dict(metadata)
        self._max_input_tokens = max_input_tokens
        self._closed = False
        self._info = BackendInfo(name="llama-cpp", model=source, revision=revision)
        self._capabilities = BackendCapabilities(min_options=2, max_options=16)

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        revision: str,
        *,
        filename: str | None = None,
        max_input_tokens: int = 4096,
        n_ctx: int | None = None,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        **llama_kwargs: Any,
    ) -> "LlamaCppBackend":
        """Load a local or Hugging Face GGUF through the optional llama.cpp runtime."""
        try:
            loaded, tokenizer, metadata = llama_cpp_backend.load_model(
                model,
                revision,
                filename=filename,
                max_input_tokens=max_input_tokens,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_gpu_layers=n_gpu_layers,
                **llama_kwargs,
            )
        except ImportError as error:
            raise ModelLoadError(
                "llama.cpp backend dependencies are missing; install 'fastjev[llama-cpp]'"
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise ModelLoadError(str(error)) from error
        return cls(loaded, tokenizer, metadata, max_input_tokens=max_input_tokens)

    @property
    def info(self) -> BackendInfo:
        return self._info

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def score(self, requests: Sequence[BackendRequest]) -> list[BackendResult]:
        if self._closed:
            raise RuntimeError("LlamaCppBackend is closed")
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
                result = llama_cpp_backend.score(
                    self._model,
                    self._tokenizer,
                    row,
                    self._metadata,
                    self._max_input_tokens,
                )
            except ValueError as error:
                if "input tokens exceed limit" in str(error):
                    raise InputTooLongError(str(error)) from error
                raise ValidationError(str(error)) from error
            except (OSError, RuntimeError) as error:
                raise BackendExecutionError(str(error)) from error
            results.append(BackendResult(
                id=result["id"],
                option_ids=tuple(result["option_ids"]),
                probabilities=tuple(result["probabilities"]),
                input_tokens=int(result.get("input_tokens", 0)),
                output_tokens=0,
                total_seconds=result.get("total_seconds"),
                forward_seconds=result.get("forward_seconds"),
                prompt_version=result.get("prompt_version"),
                probability_status=result.get(
                    "probability_status", "conditional option score; uncalibrated"
                ),
            ))
        return results

    def close(self) -> None:
        self._closed = True
        if self._model is not None:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
        self._model = None
        self._tokenizer = None
