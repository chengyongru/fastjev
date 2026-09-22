"""Workload-scoped post-hoc calibration for FastJev decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .backends.base import BackendInfo
from .errors import ValidationError


def _nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a nonempty string")


def _normalized(probabilities: Sequence[float]) -> tuple[float, ...]:
    values = tuple(probabilities)
    if len(values) < 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValidationError("probabilities must contain at least two finite nonnegative values")
    total = sum(values)
    if total <= 0:
        raise ValidationError("probabilities must contain positive probability mass")
    return tuple(float(value / total) for value in values)


def _temperature_scaled(probabilities: Sequence[float], temperature: float) -> tuple[float, ...]:
    values = _normalized(probabilities)
    inverse = 1.0 / temperature
    logs = [math.log(value) * inverse if value else -math.inf for value in values]
    maximum = max(logs)
    weights = [math.exp(value - maximum) if math.isfinite(value) else 0.0 for value in logs]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


@dataclass(frozen=True)
class CalibrationSample:
    """One labeled option distribution used to fit a temperature."""

    probabilities: tuple[float, ...]
    correct_index: int

    def __post_init__(self) -> None:
        values = _normalized(self.probabilities)
        if (
            type(self.correct_index) is not int
            or not 0 <= self.correct_index < len(values)
        ):
            raise ValidationError("correct_index must identify one supplied probability")
        object.__setattr__(self, "probabilities", values)


@dataclass(frozen=True)
class TemperatureCalibration:
    """A scalar temperature bound to one workload and exact scoring identity."""

    temperature: float
    workload: str
    backend: str
    model: str
    revision: str
    prompt_version: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValidationError("temperature must be a finite number greater than zero")
        object.__setattr__(self, "temperature", float(self.temperature))
        for field in ("workload", "backend", "model", "revision", "prompt_version"):
            _nonempty(getattr(self, field), field)

    @classmethod
    def fit(
        cls,
        samples: Iterable[CalibrationSample],
        *,
        workload: str,
        backend: str,
        model: str,
        revision: str,
        prompt_version: str,
        bounds: tuple[float, float] = (0.05, 20.0),
        iterations: int = 60,
    ) -> "TemperatureCalibration":
        """Fit one NLL-minimizing scalar without changing option ordering."""
        rows = tuple(samples)
        if not rows or any(not isinstance(row, CalibrationSample) for row in rows):
            raise ValidationError("samples must contain at least one CalibrationSample")
        if (
            len(bounds) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in bounds
            )
            or bounds[0] <= 0
            or bounds[0] >= bounds[1]
        ):
            raise ValidationError("bounds must be two increasing finite positive numbers")
        if type(iterations) is not int or iterations < 1:
            raise ValidationError("iterations must be a positive integer")

        def mean_nll(temperature: float) -> float:
            total = 0.0
            for row in rows:
                probability = _temperature_scaled(row.probabilities, temperature)[row.correct_index]
                total -= math.log(max(probability, 1e-300))
            return total / len(rows)

        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        low, high = float(bounds[0]), float(bounds[1])
        left = high - ratio * (high - low)
        right = low + ratio * (high - low)
        left_loss, right_loss = mean_nll(left), mean_nll(right)
        for _ in range(iterations):
            if left_loss < right_loss:
                high, right, right_loss = right, left, left_loss
                left = high - ratio * (high - low)
                left_loss = mean_nll(left)
            else:
                low, left, left_loss = left, right, right_loss
                right = low + ratio * (high - low)
                right_loss = mean_nll(right)
        return cls(
            temperature=(low + high) / 2.0,
            workload=workload,
            backend=backend,
            model=model,
            revision=revision,
            prompt_version=prompt_version,
        )

    def validate_backend(self, info: BackendInfo) -> None:
        """Reject a profile fitted for a different runtime or checkpoint."""
        expected = (self.backend, self.model, self.revision)
        actual = (info.name, info.model, info.revision)
        if actual != expected:
            raise ValidationError(
                "calibration identity does not match backend.info: "
                f"expected {expected!r}, got {actual!r}"
            )

    def apply(
        self,
        probabilities: Sequence[float],
        *,
        prompt_version: str | None,
    ) -> tuple[float, ...]:
        """Calibrate one distribution after verifying the frozen prompt contract."""
        if prompt_version != self.prompt_version:
            raise ValidationError(
                "calibration prompt_version does not match the backend result: "
                f"expected {self.prompt_version!r}, got {prompt_version!r}"
            )
        return _temperature_scaled(probabilities, self.temperature)

