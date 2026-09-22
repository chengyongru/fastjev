import math

import fastjev
import pytest

from fastjev import (
    BackendCapabilities,
    BackendInfo,
    BackendOption,
    BackendProtocolError,
    BackendRequest,
    BackendResult,
    Boolean,
    Choice,
    FastJev,
    Level,
    Option,
    Score,
    SystemOneAdapter,
    ValidationError,
)
from fastjev.compat.wire import SystemOneService
from fastjev._runtime.direct import score


class FakeBackend:
    def __init__(self, distributions=None, *, max_options=None):
        self.info = BackendInfo("fake", "fixture/model", "fixture-revision")
        self.capabilities = BackendCapabilities(max_options=max_options)
        self.distributions = distributions or {}
        self.calls = []
        self.closed = False

    def score(self, requests):
        self.calls.append(tuple(requests))
        return [BackendResult(
            id=request.id,
            option_ids=tuple(option.id for option in request.options),
            probabilities=tuple(self.distributions.get(
                request.id,
                [1 / len(request.options)] * len(request.options),
            )),
            input_tokens=12,
            output_tokens=0,
            total_seconds=0.02,
            forward_seconds=0.01,
            prompt_version="fixture-v1",
        ) for request in requests]

    def close(self):
        self.closed = True


def test_public_sdk_exports_high_and_low_level_entrypoints():
    assert fastjev.score_direct is score
    assert fastjev.SystemOneService is SystemOneService
    assert callable(fastjev.load_causal_model)
    assert fastjev.FastJev is FastJev
    assert fastjev.ScoringBackend is not None


def test_choice_uses_a_backend_without_exposing_model_objects():
    backend = FakeBackend({"decision": [0.1, 0.7, 0.2]})
    engine = FastJev(backend)

    result = engine.decide(
        {"ticket": "Payouts failed for three days"},
        Choice("Which team?", [
            Option("billing", "Invoices and charges"),
            Option("technical", "Bugs and integrations"),
            Option("other", "Anything else"),
        ]),
    )

    assert result.value == result.selected == "technical"
    assert result.probabilities == pytest.approx({
        "billing": 0.1,
        "technical": 0.7,
        "other": 0.2,
    })
    assert result.usage.input_tokens == 12
    assert result.provenance == fastjev.Provenance(
        backend="fake",
        model="fixture/model",
        revision="fixture-revision",
        prompt_version="fixture-v1",
        probability_status="conditional option scores; uncalibrated",
    )
    assert backend.calls[0][0] == BackendRequest(
        id="decision",
        state={"ticket": "Payouts failed for three days"},
        instruction="Which team?",
        options=(
            BackendOption("billing", "Invoices and charges"),
            BackendOption("technical", "Bugs and integrations"),
            BackendOption("other", "Anything else"),
        ),
    )


def test_boolean_and_score_share_one_backend_call():
    backend = FakeBackend({"urgent": [0.8, 0.2], "severity": [0.1, 0.3, 0.6]})
    engine = FastJev(backend)
    state = {"ticket": "Payouts failed for three days"}

    results = engine.decide_many(state, {
        "urgent": Boolean("Is this urgent?"),
        "severity": Score("How severe is this?", [
            Level(0, "Minor"),
            Level(1, "Degraded"),
            Level(2, "Blocking"),
        ]),
    })

    assert len(backend.calls) == 1
    assert [request.id for request in backend.calls[0]] == ["urgent", "severity"]
    assert all(request.state is state for request in backend.calls[0])
    assert results["urgent"].value is True
    assert results["urgent"].probabilities == pytest.approx({True: 0.8, False: 0.2})
    assert results["severity"].value == pytest.approx(1.5)
    assert results["severity"].selected == 2
    assert results["severity"].probabilities == pytest.approx({0: 0.1, 1: 0.3, 2: 0.6})
    assert 0 <= results["severity"].uncertainty.normalized_entropy <= 1
    assert results["severity"].uncertainty.calibrated is False


def test_backend_limits_and_malformed_results_fail_at_the_boundary():
    limited = FakeBackend(max_options=2)
    with pytest.raises(ValidationError, match="at most 2"):
        FastJev(limited).decide("state", Choice("Pick", [
            Option("a", "A"), Option("b", "B"), Option("c", "C")
        ]))
    assert limited.calls == []

    class WrongBackend(FakeBackend):
        def score(self, requests):
            return [BackendResult("decision", ("wrong", "ids"), (0.5, 0.5))]

    with pytest.raises(BackendProtocolError, match="option IDs"):
        FastJev(WrongBackend()).decide("state", Boolean("Is it valid?"))

    malformed = FakeBackend()
    malformed.info = BackendInfo("", "fixture/model", "fixture-revision")
    with pytest.raises(BackendProtocolError, match="backend.info"):
        FastJev(malformed)


def test_question_and_state_validation_is_backend_independent():
    with pytest.raises(ValidationError, match="unique"):
        Choice("Pick", [Option("same", "A"), Option("same", "B")])
    with pytest.raises(ValidationError, match="finite"):
        Level(math.inf, "Infinite")
    with pytest.raises(ValidationError, match="state"):
        FastJev(FakeBackend()).decide({"bad": math.nan}, Boolean("Valid?"))


def test_context_manager_closes_the_injected_backend():
    backend = FakeBackend()
    with FastJev(backend) as engine:
        assert engine.backend_info.name == "fake"
    assert backend.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        engine.decide("state", Boolean("Valid?"))


def test_system_one_adapter_uses_the_same_backend_neutral_engine():
    backend = FakeBackend({"urgent": [0.8, 0.2], "team": [0.1, 0.9]})
    adapter = SystemOneAdapter(
        FastJev(backend),
        "fastjev-test",
        "Fixture model",
        "2026-09-21",
    )
    response = adapter.evaluate({
        "model": "fastjev-test",
        "state": {"ticket": "Payouts failed"},
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Billing", "technical": "Technical"},
            },
        },
    })

    assert response["answers"]["urgent"]["noul"] == pytest.approx(0.8)
    assert response["answers"]["team"]["choice"] == "technical"
    assert response["usage"] == {"input_tokens": 24, "output_tokens": 0}
    assert len(backend.calls) == 1


def test_optional_http_module_exports_app_factory():
    pytest.importorskip("fastapi")

    from fastjev.http import create_app

    assert callable(create_app)
