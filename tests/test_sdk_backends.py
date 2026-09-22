import json

import pytest

from fastjev import BackendOption, BackendRequest
from fastjev.backends import MLXBackend, TorchBackend
from fastjev.errors import InputTooLongError, ModelLoadError
from fastjev._runtime import mlx as mlx_backend


METADATA = {"source": "fixture/model", "revision": "fixture-revision"}
REQUEST = BackendRequest(
    "decision",
    {"message": "hello"},
    "Choose",
    (BackendOption("a", "A"), BackendOption("b", "B")),
)


def scorer_result(row):
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": [0.25, 0.75],
        "input_tokens": 9,
        "total_seconds": 0.02,
        "forward_seconds": 0.01,
        "prompt_version": "direct-options-v1",
        "probability_status": "conditional option score; uncalibrated",
    }


def test_torch_backend_translates_the_public_protocol(monkeypatch):
    observed = {}

    def fake_score(model, tokenizer, row, metadata, max_tokens):
        observed.update(model=model, tokenizer=tokenizer, row=row, metadata=metadata, max_tokens=max_tokens)
        return scorer_result(row)

    monkeypatch.setattr("fastjev.backends.torch.score_direct", fake_score)
    backend = TorchBackend("model-object", "tokenizer-object", METADATA, max_input_tokens=123)
    result = backend.score([REQUEST])[0]

    assert observed == {
        "model": "model-object",
        "tokenizer": "tokenizer-object",
        "row": {
            "id": "decision",
            "state": {"message": "hello"},
            "question": "Choose",
            "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
        },
        "metadata": METADATA,
        "max_tokens": 123,
    }
    assert result.probabilities == (0.25, 0.75)
    assert backend.info.name == "torch"
    assert backend.capabilities.max_options == 16


def test_mlx_backend_implements_the_same_protocol(monkeypatch):
    observed = {}

    def fake_score(model, tokenizer, row, metadata, max_tokens):
        observed.update(row=row, max_tokens=max_tokens)
        return scorer_result(row)

    monkeypatch.setattr("fastjev.backends.mlx.mlx.score", fake_score)
    backend = MLXBackend("model-object", "tokenizer-object", METADATA, max_input_tokens=321)
    result = backend.score([REQUEST])[0]

    assert observed["row"]["question"] == "Choose"
    assert observed["max_tokens"] == 321
    assert result.option_ids == ("a", "b")
    assert backend.info.name == "mlx"


def test_backend_maps_input_limit_errors(monkeypatch):
    def reject(*_args):
        raise ValueError("Row decision: 5000 input tokens exceed limit 4096; no truncation allowed")

    monkeypatch.setattr("fastjev.backends.torch.score_direct", reject)
    backend = TorchBackend("model-object", "tokenizer-object", METADATA)
    with pytest.raises(InputTooLongError, match="5000 input tokens"):
        backend.score([REQUEST])


def test_torch_backend_reports_its_optional_dependency(monkeypatch):
    def missing(*_args):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr("fastjev.backends.torch.load_causal_model", missing)
    with pytest.raises(ModelLoadError, match=r"fastjev\[torch\]"):
        TorchBackend.from_pretrained("fixture/model", "fixture-revision")


def test_mlx_backend_accepts_only_the_validated_mlx_lm_source(monkeypatch):
    class Distribution:
        def __init__(self, commit):
            self.commit = commit

        def read_text(self, _name):
            return json.dumps({"vcs_info": {"commit_id": self.commit}})

    monkeypatch.setattr(
        mlx_backend, "distribution", lambda _name: Distribution(mlx_backend.MLX_LM_COMMIT)
    )
    assert mlx_backend._validated_mlx_lm_source()["vcs_info"]["commit_id"] == (
        mlx_backend.MLX_LM_COMMIT
    )

    monkeypatch.setattr(mlx_backend, "distribution", lambda _name: Distribution("older"))
    with pytest.raises(RuntimeError, match=mlx_backend.MLX_LM_COMMIT):
        mlx_backend._validated_mlx_lm_source()
