import pytest

from fastjev import Boolean, Choice, FastJev, Level, Option, Score, TemperatureCalibration, ValidationError
from fastjev.backends import TorchBackend
from fastjev.errors import BackendProtocolError, InputTooLongError
from test_sdk import FakeBackend


QUESTIONS = {
    "team:0": Choice("Which team?", [Option("a", "Access"), Option("b", "Billing")]),
    "urgent": Boolean("Is it urgent?"),
    "severity": Score("Severity?", [Level(0, "Minor"), Level(2, "Blocking")]),
}


def test_batch_retains_state_and_question_order_and_typed_results():
    backend = FakeBackend({"0:urgent": [0.9, 0.1], "1:urgent": [0.2, 0.8],
                           "1:severity": [0.25, 0.75]})
    states = ["first", {"message": "second"}]
    result = FastJev(backend).decide_batch(states, QUESTIONS)
    assert len(backend.calls) == 1
    assert [r.state for r in backend.calls[0]] == [states[0]] * 3 + [states[1]] * 3
    assert len({r.id for r in backend.calls[0]}) == 6
    assert [list(item) for item in result] == [list(QUESTIONS)] * 2
    assert result[0]["urgent"].value is True
    assert result[1]["urgent"].value is False
    assert result[1]["severity"].value == pytest.approx(1.5)
    assert result[1]["severity"].selected == 2
    for item in result:
        for key, decision in item.items():
            assert decision.id == key
            assert decision.provenance.model == "fixture/model"
            assert decision.usage.output_tokens == 0


@pytest.mark.parametrize("states", ["not a batch", {"state": "x"}, ["valid", ""],
                                    ["valid", {"bad": float("nan")}], iter(["x"])])
def test_batch_validates_all_states_before_backend_call(states):
    backend = FakeBackend()
    with pytest.raises(ValidationError):
        FastJev(backend).decide_batch(states, QUESTIONS)
    assert backend.calls == []


def test_empty_batch_schema_validation_and_closed_engine():
    backend = FakeBackend()
    engine = FastJev(backend)
    assert engine.decide_batch([], QUESTIONS) == []
    with pytest.raises(ValidationError):
        engine.decide_batch([], {})
    assert backend.calls == []
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.decide_batch([], QUESTIONS)


def test_batch_does_not_hide_backend_order_errors():
    class ReversedBackend(FakeBackend):
        def score(self, requests):
            return super().score(requests)[::-1]

    with pytest.raises(BackendProtocolError, match="ID"):
        FastJev(ReversedBackend()).decide_batch(["one", "two"], QUESTIONS)


def test_batch_uses_the_same_calibration_as_single_state_calls():
    calibration = TemperatureCalibration(2, "fixture", "fake", "fixture/model",
                                         "fixture-revision", "fixture-v1")
    engine = FastJev(FakeBackend({"0:urgent": [0.9, 0.1], "urgent": [0.9, 0.1]}),
                     calibration=calibration)
    single = engine.decide_many("state", {"urgent": QUESTIONS["urgent"]})
    assert engine.decide_batch(["state"], {"urgent": QUESTIONS["urgent"]}) == [single]


class ByteTokenizer:
    pad_token_id = None
    eos_token_id = 0

    def apply_chat_template(self, turns, **kwargs):
        return str(turns) + "\nANSWER:"

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


@pytest.fixture
def tiny_model():
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(123)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=256, n_positions=1024, n_embd=16, n_layer=1, n_head=2,
        attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
    )).eval()


@pytest.mark.parametrize("sort_by_length", [False, True])
def test_real_causal_model_batch_matches_serial_with_padding_and_multiple_windows(tiny_model, sort_by_length):
    metadata = {"source": "tiny", "revision": "seed-123"}
    tokenizer = ByteTokenizer()
    states = ["state " + "x" * (i * 7 % 61) for i in range(19)]
    serial = FastJev(TorchBackend(tiny_model, tokenizer, metadata))
    batched = FastJev(TorchBackend(tiny_model, tokenizer, metadata,
                                  batch_size=2, sort_by_length=sort_by_length))
    questions = {"answer": QUESTIONS["team:0"], "three": Choice("Pick", [
        Option("x", "X"), Option("y", "Y"), Option("z", "Z"),
    ]), "urgent": QUESTIONS["urgent"]}
    expected = [serial.decide_many(state, questions) for state in states]
    calls = []
    handle = tiny_model.register_forward_pre_hook(lambda model, args, kwargs: calls.append(
        tuple(kwargs["input_ids"].shape)), with_kwargs=True)
    try:
        actual = batched.decide_batch(states, questions)
    finally:
        handle.remove()
    assert len(calls) == 29
    assert [shape[0] for shape in calls] == [2] * 28 + [1]
    for baseline, result in zip(expected, actual):
        for key in questions:
            assert result[key].probabilities == pytest.approx(baseline[key].probabilities, abs=1e-6)
            assert result[key].selected == baseline[key].selected
            assert result[key].usage == baseline[key].usage
            assert result[key].provenance == baseline[key].provenance
    assert tokenizer.pad_token_id is None


def test_torch_batch_rejects_long_prompt_before_forward(tiny_model):
    engine = FastJev(TorchBackend(tiny_model, ByteTokenizer(),
                                {"source": "tiny", "revision": "seed-123"},
                                batch_size=2, max_input_tokens=600))
    calls = []
    handle = tiny_model.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        with pytest.raises(InputTooLongError, match="no truncation"):
            engine.decide_batch(["short", "x" * 800], {"answer": Boolean("Valid?")})
    finally:
        handle.remove()
    assert calls == []


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": True},
                                     {"batch_size": 1.5}, {"sort_by_length": 1}])
def test_torch_batch_options_are_validated(kwargs):
    with pytest.raises(ValidationError):
        TorchBackend(None, None, {"source": "tiny", "revision": "seed-123"}, **kwargs)
