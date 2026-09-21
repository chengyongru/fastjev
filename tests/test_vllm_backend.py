import math
from types import SimpleNamespace
import sys

import pytest

from fastjev import BackendOption, BackendRequest
from fastjev.backends import VLLMBackend
from fastjev.errors import BackendProtocolError, InputTooLongError, ModelLoadError, ValidationError


METADATA = {"source": "fixture/model", "revision": "fixture-revision"}
REQUESTS = [
    BackendRequest(
        "boolean",
        {"command": "rm -rf /tmp/cache"},
        "Is this destructive?",
        (BackendOption("true", "Yes"), BackendOption("false", "No")),
    ),
    BackendRequest(
        "action",
        {"command": "rm -rf /tmp/cache"},
        "What should happen?",
        (
            BackendOption("allow", "Allow"),
            BackendOption("review", "Review"),
            BackendOption("block", "Block"),
        ),
    ),
]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, params, use_tqdm):
        self.calls.append((prompts, params, use_tqdm))
        outputs = []
        for index, sampling in enumerate(params):
            slots = sampling.kwargs["allowed_token_ids"]
            weights = ([0.8, 0.2] if index == 0 else [0.1, 0.3, 0.6])
            position = {
                token: SimpleNamespace(logprob=math.log(weight))
                for token, weight in zip(slots, weights)
            }
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[position])]))
        return outputs


@pytest.fixture
def fake_vllm(monkeypatch):
    module = SimpleNamespace(SamplingParams=FakeSamplingParams, __version__="fixture-version")
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


def test_vllm_backend_batches_requests_and_reads_exact_option_logprobs(monkeypatch, fake_vllm):
    slots = iter(([101, 102], [101, 102, 103]))
    monkeypatch.setattr(
        "fastjev.backends.vllm.encode_prompt",
        lambda tokenizer, row, max_tokens: ([1, 2, 3], next(slots), "prompt-hash"),
    )
    engine = FakeEngine()
    backend = VLLMBackend(engine, "tokenizer", METADATA, max_input_tokens=123)

    results = backend.score(REQUESTS)

    assert len(engine.calls) == 1
    prompts, params, use_tqdm = engine.calls[0]
    assert prompts == [{"prompt_token_ids": [1, 2, 3]}] * 2
    assert use_tqdm is False
    assert params[0].kwargs == {
        "max_tokens": 1,
        "temperature": 1.0,
        "allowed_token_ids": [101, 102],
        "logprob_token_ids": [101, 102],
        "detokenize": False,
        "seed": 0,
    }
    assert params[1].kwargs["allowed_token_ids"] == [101, 102, 103]
    assert results[0].probabilities == pytest.approx((0.8, 0.2))
    assert results[1].probabilities == pytest.approx((0.1, 0.3, 0.6))
    assert results[0].input_tokens == 3
    assert results[0].output_tokens == 1
    assert results[0].prompt_version == "direct-options-v1"
    assert backend.info.name == "vllm"
    assert backend.capabilities.max_options == 16


def test_vllm_backend_maps_input_limit_errors(monkeypatch, fake_vllm):
    def reject(*_args):
        raise ValueError("Row boolean: 5000 input tokens exceed limit 4096; no truncation allowed")

    monkeypatch.setattr("fastjev.backends.vllm.encode_prompt", reject)
    backend = VLLMBackend(FakeEngine(), "tokenizer", METADATA)
    with pytest.raises(InputTooLongError, match="5000 input tokens"):
        backend.score([REQUESTS[0]])


def test_vllm_backend_rejects_missing_option_logprobs(monkeypatch, fake_vllm):
    monkeypatch.setattr(
        "fastjev.backends.vllm.encode_prompt",
        lambda *_args: ([1, 2, 3], [101, 102], "prompt-hash"),
    )

    class IncompleteEngine:
        def generate(self, *_args, **_kwargs):
            position = {101: SimpleNamespace(logprob=math.log(0.8))}
            return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[position])])]

    backend = VLLMBackend(IncompleteEngine(), "tokenizer", METADATA)
    with pytest.raises(BackendProtocolError, match="missing requested option logprobs"):
        backend.score([REQUESTS[0]])


def test_vllm_from_pretrained_pins_remote_model_and_engine_limits(monkeypatch, fake_vllm):
    observed = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def get_tokenizer(self):
            return "tokenizer"

    fake_vllm.LLM = FakeLLM
    revision = "a" * 40
    backend = VLLMBackend.from_pretrained(
        "fixture/remote-model",
        revision,
        max_input_tokens=2048,
        gpu_memory_utilization=0.5,
    )

    assert observed == {
        "model": "fixture/remote-model",
        "revision": revision,
        "tokenizer_revision": revision,
        "trust_remote_code": False,
        "max_model_len": 2049,
        "gpu_memory_utilization": 0.5,
    }
    assert backend.info.revision == revision


def test_vllm_from_pretrained_requires_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ModelLoadError, match=r"fastjev\[vllm\]"):
        VLLMBackend.from_pretrained("fixture/remote-model", "a" * 40)


def test_vllm_from_pretrained_rejects_unpinned_remote_models(fake_vllm):
    with pytest.raises(ValidationError, match="40-character"):
        VLLMBackend.from_pretrained("fixture/remote-model", "main")
