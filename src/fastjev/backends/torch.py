"""Torch direct-logit backend for the public fastjev SDK."""

from __future__ import annotations

from typing import Sequence

from .._runtime.core import load_causal_model
from .._runtime.direct import score as score_direct

from ..errors import InputTooLongError, ModelLoadError, ValidationError
from .base import (
    BackendCapabilities,
    BackendInfo,
    BackendRequest,
    BackendResult,
)


class TorchBackend:
    """Score categorical requests with one resident Transformers model."""

    def __init__(self, model, tokenizer, metadata: dict, *, max_input_tokens: int = 4096):
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValidationError("max_input_tokens must be a positive integer")
        source = metadata.get("source")
        revision = metadata.get("revision")
        if not isinstance(source, str) or not source or not isinstance(revision, str) or not revision:
            raise ValidationError("Torch metadata must contain nonempty source and revision strings")
        self._model = model
        self._tokenizer = tokenizer
        self._metadata = dict(metadata)
        self._max_input_tokens = max_input_tokens
        self._closed = False
        self._info = BackendInfo(name="torch", model=source, revision=revision)
        self._capabilities = BackendCapabilities(min_options=2, max_options=16)

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        revision: str,
        *,
        max_input_tokens: int = 4096,
    ) -> "TorchBackend":
        """Load one pinned Transformers checkpoint on the visible CUDA GPU."""
        try:
            loaded, tokenizer, metadata = load_causal_model(model, revision)
        except ImportError as error:
            raise ModelLoadError(
                "Torch backend dependencies are missing; install 'fastjev[torch]'"
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
            raise RuntimeError("TorchBackend is closed")
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
                result = score_direct(
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
