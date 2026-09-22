import pytest

from fastjev import (
    BackendCapabilities,
    BackendInfo,
    BackendResult,
    Boolean,
    CalibrationSample,
    FastJev,
    TemperatureCalibration,
    ValidationError,
)


class Backend:
    info = BackendInfo("fixture", "fixture/model", "a" * 40)
    capabilities = BackendCapabilities()

    def score(self, requests):
        return [BackendResult(
            id=request.id,
            option_ids=tuple(option.id for option in request.options),
            probabilities=(0.9, 0.1),
            prompt_version="direct-options-v1",
        ) for request in requests]

    def close(self):
        pass


def profile(temperature=2.0):
    return TemperatureCalibration(
        temperature=temperature,
        workload="support-routing-v1",
        backend="fixture",
        model="fixture/model",
        revision="a" * 40,
        prompt_version="direct-options-v1",
    )


def test_calibration_is_applied_and_audited_without_changing_argmax():
    result = FastJev(Backend(), calibration=profile()).decide(
        "evidence", Boolean("Does it apply?")
    )

    assert result.selected is True
    assert result.probabilities[True] == pytest.approx(0.75)
    assert result.uncertainty.calibrated is True
    assert result.calibration is not None
    assert result.calibration.temperature == 2.0
    assert result.calibration.workload == "support-routing-v1"
    assert "temperature calibrated" in result.provenance.probability_status


def test_calibration_rejects_backend_and_prompt_identity_mismatches():
    with pytest.raises(ValidationError, match="identity"):
        FastJev(Backend(), calibration=TemperatureCalibration(
            2.0,
            "support-routing-v1",
            "other-backend",
            "fixture/model",
            "a" * 40,
            "direct-options-v1",
        ))

    wrong_prompt = profile()
    wrong_prompt = TemperatureCalibration(
        wrong_prompt.temperature,
        wrong_prompt.workload,
        wrong_prompt.backend,
        wrong_prompt.model,
        wrong_prompt.revision,
        "other-prompt",
    )
    with pytest.raises(ValidationError, match="prompt_version"):
        FastJev(Backend(), calibration=wrong_prompt).decide(
            "evidence", Boolean("Does it apply?")
        )


def test_fit_returns_a_valid_identity_bound_profile_and_preserves_order():
    samples = [
        CalibrationSample((0.99, 0.01), 1),
        CalibrationSample((0.95, 0.05), 1),
        CalibrationSample((0.9, 0.1), 0),
    ]
    fitted = TemperatureCalibration.fit(
        samples,
        workload="fixture",
        backend="fixture",
        model="fixture/model",
        revision="a" * 40,
        prompt_version="direct-options-v1",
    )

    assert fitted.temperature > 1
    calibrated = fitted.apply((0.8, 0.15, 0.05), prompt_version="direct-options-v1")
    assert calibrated[0] > calibrated[1] > calibrated[2]
    assert sum(calibrated) == pytest.approx(1.0)


@pytest.mark.parametrize("temperature", [0, -1, float("inf"), float("nan")])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValidationError, match="temperature"):
        profile(temperature)

