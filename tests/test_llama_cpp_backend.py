import hashlib
import sys
from types import SimpleNamespace

import pytest

from fastjev import BackendOption, BackendRequest
from fastjev.backends import LlamaCppBackend
from fastjev.errors import ModelLoadError, ValidationError
from semif_phase1 import llama_cpp_backend


METADATA = {"source": "fixture/model.gguf", "revision": "fixture-revision"}
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


def test_llama_cpp_backend_translates_the_public_protocol(monkeypatch):
    observed = {}

    def fake_score(model, tokenizer, row, metadata, max_tokens):
        observed.update(model=model, tokenizer=tokenizer, row=row, metadata=metadata, max_tokens=max_tokens)
        return scorer_result(row)

    monkeypatch.setattr("fastjev.backends.llama_cpp.llama_cpp_backend.score", fake_score)
    backend = LlamaCppBackend("model-object", "tokenizer-object", METADATA, max_input_tokens=123)
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
    assert backend.info.name == "llama-cpp"
    assert backend.capabilities.max_options == 16


def test_llama_cpp_backend_rejects_invalid_option_prompt(monkeypatch):
    def reject(*_args):
        raise ValueError("Answer boundary changes tokenization for slot 'A'")

    monkeypatch.setattr("fastjev.backends.llama_cpp.llama_cpp_backend.score", reject)
    backend = LlamaCppBackend("model-object", "tokenizer-object", METADATA)
    with pytest.raises(ValidationError, match="Answer boundary"):
        backend.score([REQUEST])


class FakeModel:
    metadata = {}

    def __init__(self):
        self.evaluated = None

    def tokenize(self, text, *, add_bos, special):
        assert add_bos is False
        assert special is True
        prompt = text.decode("utf-8")
        values = {
            "PROMPT": [10, 11],
            "PROMPTA": [10, 11, 101],
            "PROMPTB": [10, 11, 102],
        }
        return values[prompt]

    def detokenize(self, tokens, *, special):
        assert special is False
        return {101: b"A", 102: b"B"}[tokens[0]]

    def reset(self):
        self.evaluated = None

    def eval(self, tokens):
        self.evaluated = list(tokens)

    @property
    def eval_logits(self):
        values = [0.0] * 103
        values[101] = 2.0
        values[102] = 1.0
        return [values]


class FakeFormatter:
    def __call__(self, *, messages, enable_thinking):
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert enable_thinking is False
        return SimpleNamespace(prompt="PROMPT", added_special=True)


def test_score_reads_last_position_logits_and_preserves_provenance(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(llama_cpp_backend, "_chat_formatter", lambda _model: FakeFormatter())
    row = {
        "id": "decision",
        "state": "Evidence",
        "question": "Choose",
        "options": [
            {"id": "a", "description": "A"},
            {"id": "b", "description": "B"},
        ],
    }

    result = llama_cpp_backend.score(model, None, row, METADATA, max_tokens=10)

    assert model.evaluated == [10, 11]
    assert result["answer_token_ids"] == [101, 102]
    assert result["probabilities"][0] > result["probabilities"][1]
    assert result["output_tokens"] == 0
    assert result["readout"].startswith("native llama.cpp")


def test_load_model_resolves_a_single_gguf_and_pins_runtime(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fixture-gguf")
    observed = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    fake_module = SimpleNamespace(Llama=FakeLlama, __version__="fixture-version")
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    model, tokenizer, metadata = llama_cpp_backend.load_model(
        str(model_path),
        "local-revision",
        max_input_tokens=123,
        n_batch=64,
        n_gpu_layers=7,
    )

    assert isinstance(model, FakeLlama)
    assert tokenizer is None
    assert observed == {
        "model_path": str(model_path.resolve()),
        "n_ctx": 124,
        "n_batch": 64,
        "n_gpu_layers": 7,
        "logits_all": True,
        "verbose": False,
    }
    assert metadata["source_artifact_sha256"] == hashlib.sha256(b"fixture-gguf").hexdigest()
    assert metadata["llama_cpp_version"] == "fixture-version"


def test_load_model_accepts_a_directory_with_one_gguf(monkeypatch, tmp_path):
    model_path = tmp_path / "nested.gguf"
    model_path.write_bytes(b"fixture-gguf")
    fake_module = SimpleNamespace(Llama=lambda **_kwargs: object(), __version__="fixture-version")
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    model, _, _ = llama_cpp_backend.load_model(str(tmp_path), "local-revision")
    assert model is not None


def test_load_model_grows_implicit_context_for_the_default_batch(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fixture-gguf")
    observed = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=FakeLlama, __version__="fixture-version"),
    )

    llama_cpp_backend.load_model(str(model_path), "local-revision", max_input_tokens=128)

    assert observed["n_ctx"] == 512


def test_llama_cpp_backend_reports_missing_optional_dependency(monkeypatch):
    def missing(*_args, **_kwargs):
        raise ImportError("No module named 'llama_cpp'")

    monkeypatch.setattr("fastjev.backends.llama_cpp.llama_cpp_backend.load_model", missing)
    with pytest.raises(ModelLoadError, match=r"fastjev\[llama-cpp\]"):
        LlamaCppBackend.from_pretrained("model.gguf", "local-revision")
