"""Built-in backends and the protocol for third-party implementations."""

from .base import (
    BackendCapabilities,
    BackendInfo,
    BackendOption,
    BackendRequest,
    BackendResult,
    JsonValue,
    ScoringBackend,
)
from .llama_cpp import LlamaCppBackend
from .mlx import MLXBackend
from .torch import TorchBackend
from .vllm import VLLMBackend

__all__ = [
    "BackendCapabilities",
    "BackendInfo",
    "BackendOption",
    "BackendRequest",
    "BackendResult",
    "JsonValue",
    "LlamaCppBackend",
    "MLXBackend",
    "ScoringBackend",
    "TorchBackend",
    "VLLMBackend",
]
