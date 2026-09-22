from types import SimpleNamespace
import sys

import pytest

from fastjev import BackendOption, BackendRequest, ExLlamaV3Backend
from fastjev._runtime import exl3


METADATA = {"source": "fixture/model", "revision": "fixture-revision"}
REQUEST = BackendRequest(
    "decision",
    {"message": "hello"},
    "Choose",
    (BackendOption("a", "A"), BackendOption("b", "B")),
)


def test_public_backend_translates_the_protocol(monkeypatch):
    observed = {}

    def fake_score(runtime, tokenizer, row, metadata, max_tokens):
        observed.update(
            runtime=runtime,
            tokenizer=tokenizer,
            row=row,
            metadata=metadata,
            max_tokens=max_tokens,
        )
        return {
            "id": row["id"],
            "option_ids": [option["id"] for option in row["options"]],
            "probabilities": [0.2, 0.8],
            "input_tokens": 12,
            "prompt_version": "direct-options-v1",
        }

    monkeypatch.setattr("fastjev.backends.exl3.exl3.score", fake_score)
    backend = ExLlamaV3Backend("runtime", "tokenizer", METADATA, max_input_tokens=123)
    result = backend.score([REQUEST])[0]

    assert observed["row"]["question"] == "Choose"
    assert observed["max_tokens"] == 123
    assert result.probabilities == (0.2, 0.8)
    assert backend.info.name == "exl3"


def test_token_helpers_accept_runtime_return_shapes(monkeypatch):
    class TorchTensor:
        pass

    fake_torch = SimpleNamespace(Tensor=TorchTensor, tensor=lambda values: values)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    encoded_special = []

    class Tokenizer:
        def encode(self, text, *, encode_special_tokens=False):
            encoded_special.append(encode_special_tokens)
            return ([ord(character) for character in text], text)

        def single_id(self, text):
            return ord(text)

        def decode(self, values):
            return "".join(chr(value) for value in values)

    tokenizer = Tokenizer()
    ids = exl3._encode_ids(tokenizer, "abc")
    assert ids == [97, 98, 99]
    assert exl3._slot_ids(tokenizer, 2, "abc", ids) == [65, 66]
    assert encoded_special == [True, True, True]


def test_runtime_rejects_over_budget_inputs(monkeypatch):
    class Tokenizer:
        def hf_render_chat_template(self, *_args, **_kwargs):
            return "abcd"

        def encode(self, text, *, encode_special_tokens=False):
            assert encode_special_tokens is True
            return list(range(len(text)))

    with pytest.raises(ValueError, match="cache budget"):
        exl3._encode_prompt(
            Tokenizer(),
            {
                "id": "x",
                "state": "state",
                "question": "question",
                "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
            },
            max_tokens=10,
            cache_size=4,
        )
