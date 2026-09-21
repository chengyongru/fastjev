"""Backend-neutral scoring protocol used by :class:`fastjev.FastJev`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, TypeAlias, runtime_checkable


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True)
class BackendInfo:
    """Stable model identity reported with every decision."""

    name: str
    model: str
    revision: str


@dataclass(frozen=True)
class BackendCapabilities:
    """Limits that the engine enforces before invoking a backend."""

    min_options: int = 2
    max_options: int | None = None


@dataclass(frozen=True)
class BackendOption:
    """One internal categorical option passed to a backend."""

    id: str
    description: str


@dataclass(frozen=True)
class BackendRequest:
    """The complete backend input for one categorical decision."""

    id: str
    state: JsonValue
    instruction: str
    options: tuple[BackendOption, ...]


@dataclass(frozen=True)
class BackendResult:
    """Normalized transport shape returned by every scoring backend."""

    id: str
    option_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    input_tokens: int = 0
    output_tokens: int = 0
    total_seconds: float | None = None
    forward_seconds: float | None = None
    prompt_version: str | None = None
    probability_status: str = "conditional option scores; uncalibrated"


@runtime_checkable
class ScoringBackend(Protocol):
    """Minimal protocol for Torch, MLX, vLLM, or remote implementations."""

    @property
    def info(self) -> BackendInfo: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def score(self, requests: Sequence[BackendRequest]) -> Sequence[BackendResult]: ...

    def close(self) -> None: ...
