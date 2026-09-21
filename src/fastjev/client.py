"""High-level, backend-neutral semantic decision engine."""

from __future__ import annotations

import json
import math
from threading import Lock
from typing import Mapping, Sequence

from .backends.base import (
    BackendCapabilities,
    BackendInfo,
    BackendOption,
    BackendRequest,
    BackendResult,
    JsonValue,
    ScoringBackend,
)
from .errors import BackendProtocolError, ValidationError
from .types import (
    Boolean,
    Choice,
    Decision,
    Provenance,
    Question,
    Score,
    Timing,
    Uncertainty,
    Usage,
)


class FastJev:
    """Keep one scoring backend resident and return typed semantic decisions."""

    def __init__(self, backend: ScoringBackend):
        if not isinstance(backend, ScoringBackend):
            raise TypeError("backend must implement the ScoringBackend protocol")
        info = backend.info
        capabilities = backend.capabilities
        if not isinstance(info, BackendInfo) or not all(
            isinstance(value, str) and value for value in (info.name, info.model, info.revision)
        ):
            raise BackendProtocolError(
                "backend.info must be BackendInfo with nonempty name, model, and revision"
            )
        if (
            not isinstance(capabilities, BackendCapabilities)
            or type(capabilities.min_options) is not int
            or capabilities.min_options < 2
            or (
                capabilities.max_options is not None
                and (
                    type(capabilities.max_options) is not int
                    or capabilities.max_options < capabilities.min_options
                )
            )
        ):
            raise BackendProtocolError("backend.capabilities contains invalid option limits")
        self._backend = backend
        self._info = info
        self._capabilities = capabilities
        self._lock = Lock()
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        revision: str,
        *,
        max_input_tokens: int = 4096,
    ) -> "FastJev":
        """Convenience constructor for the built-in Torch/CUDA backend."""
        from .backends.torch import TorchBackend

        return cls(TorchBackend.from_pretrained(
            model,
            revision,
            max_input_tokens=max_input_tokens,
        ))

    @property
    def backend_info(self):
        return self._info

    def decide(self, state: JsonValue, question: Question) -> Decision:
        """Evaluate one typed question against a state."""
        return self.decide_many(state, {"decision": question})["decision"]

    def decide_many(
        self,
        state: JsonValue,
        questions: Mapping[str, Question],
    ) -> dict[str, Decision]:
        """Evaluate multiple questions in one backend call while preserving IDs."""
        if self._closed:
            raise RuntimeError("FastJev is closed")
        _validate_state(state)
        if not isinstance(questions, Mapping) or not questions:
            raise ValidationError("questions must be a nonempty mapping")

        requests = []
        value_maps = []
        score_values = []
        for question_id, question in questions.items():
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValidationError("question IDs must be nonempty strings")
            request, values, numeric = _compile(question_id, state, question)
            option_count = len(request.options)
            capabilities = self._capabilities
            if option_count < capabilities.min_options:
                raise ValidationError(
                    f"question {question_id!r} requires at least {capabilities.min_options} options"
                )
            if capabilities.max_options is not None and option_count > capabilities.max_options:
                raise ValidationError(
                    f"backend {self._info.name!r} supports at most "
                    f"{capabilities.max_options} options per question"
                )
            requests.append(request)
            value_maps.append(values)
            score_values.append(numeric)

        with self._lock:
            raw_results = list(self._backend.score(requests))
        if len(raw_results) != len(requests):
            raise BackendProtocolError("backend returned a different number of results than requested")

        decisions = {}
        for request, result, values, numeric in zip(requests, raw_results, value_maps, score_values):
            probabilities = _validate_result(request, result)
            selected_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            mapped = {value: probability for value, probability in zip(values, probabilities)}
            selected = values[selected_index]
            value = selected if numeric is None else sum(
                level * probability for level, probability in zip(numeric, probabilities)
            )
            entropy = -sum(probability * math.log(probability) for probability in probabilities if probability)
            normalized_entropy = entropy / math.log(len(probabilities))
            decisions[request.id] = Decision(
                id=request.id,
                value=value,
                selected=selected,
                probabilities=mapped,
                usage=Usage(result.input_tokens, result.output_tokens),
                timing=Timing(result.total_seconds, result.forward_seconds),
                provenance=Provenance(
                    backend=self._info.name,
                    model=self._info.model,
                    revision=self._info.revision,
                    prompt_version=result.prompt_version,
                    probability_status=result.probability_status,
                ),
                uncertainty=Uncertainty(
                    normalized_entropy=max(0.0, min(1.0, normalized_entropy)),
                    calibrated=False,
                ),
            )
        return decisions

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            if not self._closed:
                try:
                    self._backend.close()
                finally:
                    self._closed = True

    def __enter__(self) -> "FastJev":
        if self._closed:
            raise RuntimeError("FastJev is closed")
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def _validate_state(state: JsonValue) -> None:
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValidationError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValidationError("state must contain finite JSON-compatible values") from error


def _compile(
    question_id: str,
    state: JsonValue,
    question: Question,
) -> tuple[BackendRequest, tuple, tuple[float, ...] | None]:
    if isinstance(question, Choice):
        options = tuple(BackendOption(option.id, option.description) for option in question.options)
        values = tuple(option.id for option in question.options)
        numeric = None
    elif isinstance(question, Boolean):
        options = (
            BackendOption("true", question.true_description),
            BackendOption("false", question.false_description),
        )
        values = (True, False)
        numeric = None
    elif isinstance(question, Score):
        options = tuple(
            BackendOption(str(index), f"Level {index}: {level.description}")
            for index, level in enumerate(question.levels)
        )
        values = tuple(level.value for level in question.levels)
        numeric = tuple(float(level.value) for level in question.levels)
    else:
        raise ValidationError(f"question {question_id!r} has an unsupported type")
    return BackendRequest(question_id, state, question.instruction, options), values, numeric


def _validate_result(request: BackendRequest, result: BackendResult) -> list[float]:
    if not isinstance(result, BackendResult):
        raise BackendProtocolError(f"backend result for {request.id!r} has an unsupported type")
    expected_ids = tuple(option.id for option in request.options)
    if result.id != request.id or result.option_ids != expected_ids:
        raise BackendProtocolError(
            f"backend result for {request.id!r} does not match the requested option IDs"
        )
    values = result.probabilities
    if len(values) != len(expected_ids):
        raise BackendProtocolError(f"backend result for {request.id!r} has the wrong probability count")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise BackendProtocolError(
            f"backend result for {request.id!r} contains invalid probabilities"
        )
    total = sum(values)
    if total <= 0:
        raise BackendProtocolError(f"backend result for {request.id!r} has zero probability mass")
    if type(result.input_tokens) is not int or result.input_tokens < 0:
        raise BackendProtocolError(f"backend result for {request.id!r} has invalid input_tokens")
    if type(result.output_tokens) is not int or result.output_tokens < 0:
        raise BackendProtocolError(f"backend result for {request.id!r} has invalid output_tokens")
    if result.prompt_version is not None and (
        not isinstance(result.prompt_version, str) or not result.prompt_version
    ):
        raise BackendProtocolError(f"backend result for {request.id!r} has invalid prompt_version")
    if not isinstance(result.probability_status, str) or not result.probability_status:
        raise BackendProtocolError(f"backend result for {request.id!r} has invalid probability_status")
    _validate_timing(request.id, "total_seconds", result.total_seconds)
    _validate_timing(request.id, "forward_seconds", result.forward_seconds)
    return [float(value / total) for value in values]


def _validate_timing(question_id: str, field: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise BackendProtocolError(f"backend result for {question_id!r} has invalid {field}")
