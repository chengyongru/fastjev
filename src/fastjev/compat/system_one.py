"""TypeSafe System One wire adapter for a :class:`fastjev.FastJev` engine."""

from __future__ import annotations

from .wire import (
    SystemOneValidationError,
    request_rows,
    response_from_results,
)

from ..client import FastJev
from ..errors import ValidationError
from ..types import Choice, Option


class SystemOneAdapter:
    """Expose System One request and response shapes without coupling the engine."""

    def __init__(
        self,
        engine: FastJev,
        served_model: str,
        description: str,
        release_date: str,
    ):
        if not all(isinstance(value, str) and value for value in (
            served_model,
            description,
            release_date,
        )):
            raise ValueError("served model metadata must contain nonempty strings")
        self.engine = engine
        self.served_model = served_model
        self.description = description
        self.release_date = release_date

    def evaluate(self, payload: dict) -> dict:
        specs, rows = request_rows(payload, self.served_model)
        questions = {
            row["id"]: Choice(
                row["question"],
                tuple(Option(option["id"], option["description"]) for option in row["options"]),
            )
            for row in rows
        }
        try:
            decisions = self.engine.decide_many(payload["state"], questions)
        except ValidationError as error:
            raise SystemOneValidationError(f"questions: {error}") from error
        results = []
        for spec in specs:
            decision = decisions[spec.id]
            results.append({
                "id": spec.id,
                "option_ids": list(spec.option_ids),
                "probabilities": [decision.probabilities[option_id] for option_id in spec.option_ids],
                "input_tokens": decision.usage.input_tokens,
                "prompt_version": decision.provenance.prompt_version,
                "probability_status": decision.provenance.probability_status,
                "calibration": (
                    None
                    if decision.calibration is None
                    else {
                        "method": decision.calibration.method,
                        "temperature": decision.calibration.temperature,
                        "workload": decision.calibration.workload,
                    }
                ),
            })
        return response_from_results(self.served_model, specs, results)

    def models(self) -> dict:
        return {"models": [{
            "name": self.served_model,
            "description": self.description,
            "release_date": self.release_date,
        }]}
