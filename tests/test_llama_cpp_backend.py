import hashlib
import sys
from types import SimpleNamespace

import pytest

from fastjev import BackendOption, BackendRequest
from fastjev.backends import LlamaCppBackend
from fastjev.errors import ModelLoadError, ValidationError
from fastjev._runtime import llama_cpp as llama_cpp_backend


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

    monkeypatch.setattr("fastjev.backends.llama_cpp.llama_cpp.score", fake_score)
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


def test_llama_cpp_backend_uses_explicit_shared_prefix_path(monkeypatch):
    observed = {}

    def fake_shared(model, tokenizer, rows, metadata, max_tokens):
        observed.update(rows=rows, max_tokens=max_tokens)
        return [scorer_result(row) for row in rows], {"prefix_tokens": 3}

    monkeypatch.setattr("fastjev.backends.llama_cpp.llama_cpp.score_shared", fake_shared)
    backend = LlamaCppBackend(
        "model-object",
        "tokenizer-object",
        METADATA,
        max_input_tokens=123,
        prefix_reuse=True,
    )

    results = backend.score([REQUEST, BackendRequest(
        "other",
        REQUEST.state,
        REQUEST.instruction,
        REQUEST.options,
    )])

    assert [result.id for result in results] == ["decision", "other"]
    assert observed["max_tokens"] == 123
    assert observed["rows"][0]["state"] == observed["rows"][1]["state"]


def test_llama_cpp_shared_runtime_restores_one_saved_prefix(monkeypatch):
    class State:
        llama_state = b"snapshot"

    class Model:
        def __init__(self):
            self.events = []
            self._logits = None

        def reset(self):
            self.events.append("reset")

        def eval(self, tokens):
            self.events.append(("eval", list(tokens)))
            values = [0.0] * 103
            if tokens == [10]:
                values[101], values[102] = 3.0, 1.0
            elif tokens == [20]:
                values[101], values[102] = 1.0, 3.0
            self._logits = [values]

        def save_state(self):
            self.events.append("save")
            return State()

        def load_state(self, _state):
            self.events.append("load")

        @property
        def eval_logits(self):
            return self._logits

    rows = [
        {"id": "a", "state": "same", "question": "a", "options": [{"id": "yes"}, {"id": "no"}]},
        {"id": "b", "state": "same", "question": "b", "options": [{"id": "yes"}, {"id": "no"}]},
    ]
    encoded = {
        "a": ([1, 2, 10], [101, 102], "hash-a"),
        "b": ([1, 2, 20], [101, 102], "hash-b"),
    }
    monkeypatch.setattr(
        llama_cpp_backend,
        "_encode_prompt",
        lambda _model, row, _max: encoded[row["id"]],
    )
    monkeypatch.setattr(llama_cpp_backend, "_state_prefix", lambda _model, _state: [1, 2])
    model = Model()

    results, timing = llama_cpp_backend.score_shared(model, None, rows, METADATA, 10)

    assert model.events.count("save") == 1
    assert model.events.count("load") == 2
    assert results[0]["probabilities"][0] > results[0]["probabilities"][1]
    assert results[1]["probabilities"][1] > results[1]["probabilities"][0]
    assert timing["prefix_tokens"] == 2


def test_llama_cpp_backend_rejects_invalid_option_prompt(monkeypatch):
    def reject(*_args):
        raise ValueError("Answer boundary changes tokenization for slot 'A'")

    monkeypatch.setattr("fastjev.backends.llama_cpp.llama_cpp.score", reject)
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
    assert metadata["source"] == str(model_path.resolve())
    assert metadata["source_filename"] is None
    assert metadata["resolved_model_path"] == str(model_path.resolve())


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


def test_load_model_downloads_an_exact_remote_gguf(monkeypatch, tmp_path):
    model_path = tmp_path / "model-q4.gguf"
    model_path.write_bytes(b"remote-fixture-gguf")
    download = {}
    loaded = {}

    def fake_download(**kwargs):
        download.update(kwargs)
        return str(model_path)

    class FakeLlama:
        def __init__(self, **kwargs):
            loaded.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=FakeLlama, __version__="fixture-version"),
    )
    revision = "a" * 40

    _, _, metadata = llama_cpp_backend.load_model(
        "fixture/repo",
        revision,
        filename="quantized/model-q4.gguf",
    )

    assert download == {
        "repo_id": "fixture/repo",
        "filename": "quantized/model-q4.gguf",
        "revision": revision,
    }
    assert loaded["model_path"] == str(model_path.resolve())
    assert metadata["source"] == "fixture/repo"
    assert metadata["source_filename"] == "quantized/model-q4.gguf"
    assert metadata["resolved_model_path"] == str(model_path.resolve())
    assert metadata["source_artifact_sha256"] == hashlib.sha256(
        b"remote-fixture-gguf"
    ).hexdigest()


@pytest.mark.parametrize(
    "revision,filename,message",
    [
        ("main", "model.gguf", "pinned 40-character"),
        ("a" * 40, None, "exact .gguf filename"),
        ("a" * 40, "model.safetensors", "exact .gguf filename"),
    ],
)
def test_remote_gguf_requires_an_exact_file_and_revision(revision, filename, message):
    with pytest.raises(ValueError, match=message):
        llama_cpp_backend.load_model(
            "fixture/repo",
            revision,
            filename=filename,
        )


def test_local_gguf_rejects_remote_filename(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fixture-gguf")

    with pytest.raises(ValueError, match="only valid for a Hugging Face"):
        llama_cpp_backend.load_model(
            str(model_path),
            "local-revision",
            filename="model.gguf",
        )


def test_public_backend_forwards_remote_filename(monkeypatch):
    observed = {}

    def fake_load_model(source, revision, **kwargs):
        observed.update(source=source, revision=revision, **kwargs)
        return object(), None, {"source": source, "revision": revision}

    monkeypatch.setattr(
        "fastjev.backends.llama_cpp.llama_cpp.load_model",
        fake_load_model,
    )
    revision = "b" * 40

    backend = LlamaCppBackend.from_pretrained(
        "fixture/repo",
        revision,
        filename="model-q4.gguf",
    )

    assert observed["filename"] == "model-q4.gguf"
    assert backend.info.model == "fixture/repo"


def test_public_backend_rejects_remote_without_filename():
    with pytest.raises(ModelLoadError, match="exact .gguf filename"):
        LlamaCppBackend.from_pretrained("fixture/repo", "a" * 40)


def test_llama_cpp_backend_reports_missing_optional_dependency(monkeypatch):
    def missing(*_args, **_kwargs):
        raise ImportError("No module named 'llama_cpp'")

    monkeypatch.setattr("fastjev.backends.llama_cpp.llama_cpp.load_model", missing)
    with pytest.raises(ModelLoadError, match=r"fastjev\[llama-cpp\]"):
        LlamaCppBackend.from_pretrained("model.gguf", "local-revision")
