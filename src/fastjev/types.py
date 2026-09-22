"""Typed questions and results for backend-neutral semantic decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Generic, Mapping, TypeAlias, TypeVar

from .errors import ValidationError


T = TypeVar("T")


def _nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a nonempty string")


@dataclass(frozen=True)
class Option:
    """A stable choice ID and its natural-language meaning."""

    id: str
    description: str

    def __post_init__(self) -> None:
        _nonempty(self.id, "option.id")
        _nonempty(self.description, f"option {self.id!r} description")


@dataclass(frozen=True)
class Level:
    """One numeric level in an ordered score question."""

    value: float
    description: str

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
            raise ValidationError("level.value must be a finite number")
        _nonempty(self.description, "level.description")


@dataclass(frozen=True)
class Choice:
    """Choose one declared option."""

    instruction: str
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        _nonempty(self.instruction, "choice.instruction")
        object.__setattr__(self, "options", tuple(self.options))
        if len(self.options) < 2:
            raise ValidationError("choice.options must contain at least two options")
        if any(not isinstance(option, Option) for option in self.options):
            raise ValidationError("choice.options must contain Option values")
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)):
            raise ValidationError("choice option IDs must be unique")


@dataclass(frozen=True)
class Boolean:
    """Decide whether a natural-language condition is true."""

    instruction: str
    true_description: str = "The condition is true."
    false_description: str = "The condition is false."

    def __post_init__(self) -> None:
        _nonempty(self.instruction, "boolean.instruction")
        _nonempty(self.true_description, "boolean.true_description")
        _nonempty(self.false_description, "boolean.false_description")


@dataclass(frozen=True)
class Score:
    """Choose an ordered level and return its probability-weighted value."""

    instruction: str
    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        _nonempty(self.instruction, "score.instruction")
        object.__setattr__(self, "levels", tuple(self.levels))
        if len(self.levels) < 2:
            raise ValidationError("score.levels must contain at least two levels")
        if any(not isinstance(level, Level) for level in self.levels):
            raise ValidationError("score.levels must contain Level values")
        values = [level.value for level in self.levels]
        if len(values) != len(set(values)):
            raise ValidationError("score level values must be unique")


Question: TypeAlias = Boolean | Choice | Score


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Timing:
    total_seconds: float | None
    forward_seconds: float | None


@dataclass(frozen=True)
class Provenance:
    backend: str
    model: str
    revision: str
    prompt_version: str | None
    probability_status: str


@dataclass(frozen=True)
class Uncertainty:
    """Distribution entropy; zero is peaked and one is uniform."""

    normalized_entropy: float
    calibrated: bool = False


@dataclass(frozen=True)
class Calibration:
    """Calibration applied to the returned option distribution."""

    method: str
    temperature: float
    workload: str


@dataclass(frozen=True)
class Decision(Generic[T]):
    """A typed decision plus distribution, resource use, and provenance."""

    id: str
    value: T | float
    selected: T
    probabilities: Mapping[T, float]
    usage: Usage
    timing: Timing
    provenance: Provenance
    uncertainty: Uncertainty
    calibration: Calibration | None = None
