"""MLX direct-logit backend for the public fastjev SDK."""

from __future__ import annotations

from typing import Sequence

from semif_phase1 import mlx_backend

from ..errors import InputTooLongError, ModelLoadError, ValidationError
from .base import BackendCapabilities, BackendInfo, BackendRequest, BackendResult


class MLXBackend:
    """Score categorical requests with one resident MLX-LM model."""

    def __init__(self, model, tokenizer, metadata: dict, *, max_input_tokens: int = 4096):
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValidationError("max_input_tokens must be a positive integer")
        source = metadata.get("source")
        revision = metadata.get("revision")
        if not isinstance(source, str) or not source or not isinstance(revision, str) or not revision:
            raise ValidationError("MLX metadata must contain nonempty source and revision strings")
        self._model = model
        self._tokenizer = tokenizer
        self._metadata = dict(metadata)
        self._max_input_tokens = max_input_tokens
        self._closed = False
        self._info = BackendInfo(name="mlx", model=source, revision=revision)
        self._capabilities = BackendCapabilities(min_options=2, max_options=16)

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        revision: str,
        *,
        bits: int | None = None,
        cache_limit_mib: int = mlx_backend.DEFAULT_CACHE_LIMIT_MIB,
        max_input_tokens: int = 4096,
    ) -> "MLXBackend":
        """Load one pinned native Qwen3.5 checkpoint through MLX-LM."""
        try:
            loaded, tokenizer, metadata = mlx_backend.load_model(
                model,
                revision,
                bits,
                cache_limit_mib=cache_limit_mib,
            )
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
            raise RuntimeError("MLXBackend is closed")
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
                result = mlx_backend.score(
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
                    "probability_status", "conditional option scores; uncalibrated"
                ),
            ))
        return results

    def close(self) -> None:
        self._closed = True
        self._model = None
        self._tokenizer = None
