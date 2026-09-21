"""Run one typed decision through the high-level fastjev SDK."""

import json

from fastjev import Choice, FastJev, Option


MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def main() -> None:
    with FastJev.from_pretrained(MODEL, REVISION) as jev:
        result = jev.decide(
            {"message": "The customer cannot access the account after a password reset."},
            Choice("Which queue should handle this request?", [
                Option("access", "Account access and authentication support."),
                Option("billing", "Billing, payments, and refunds."),
            ]),
        )
    print(json.dumps({
        "value": result.value,
        "probabilities": result.probabilities,
        "input_tokens": result.usage.input_tokens,
        "total_seconds": result.timing.total_seconds,
        "backend": result.provenance.backend,
        "model": result.provenance.model,
        "revision": result.provenance.revision,
        "calibrated": result.uncertainty.calibrated,
    }, indent=2))


if __name__ == "__main__":
    main()
