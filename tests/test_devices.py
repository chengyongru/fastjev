from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from fastjev._runtime.core import load_causal_model, resolve_device, synchronize


@pytest.fixture
def torch_stub(monkeypatch):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=Mock(return_value=False),
            device_count=Mock(return_value=0),
            synchronize=Mock(),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=True))),
        mps=SimpleNamespace(synchronize=Mock()),
        device=lambda name: SimpleNamespace(type=name.split(":")[0], name=name),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_auto_prefers_cuda_and_falls_back_to_mps(torch_stub):
    assert resolve_device().name == "mps"
    torch_stub.cuda.is_available.return_value = True
    torch_stub.cuda.device_count.return_value = 1
    assert resolve_device().name == "cuda:0"


def test_unavailable_or_ambiguous_devices_fail(torch_stub):
    with pytest.raises(ValueError, match="exactly one CUDA"):
        resolve_device("cuda")
    torch_stub.backends.mps.is_available.return_value = False
    with pytest.raises(ValueError, match="MPS is unavailable"):
        resolve_device("mps")
    with pytest.raises(ValueError, match="device must"):
        resolve_device("cpu")


@pytest.mark.parametrize("kind", ["cuda", "mps", "cpu"])
def test_synchronization_dispatch(torch_stub, kind):
    device = SimpleNamespace(type=kind)
    synchronize(device)
    assert torch_stub.cuda.synchronize.call_count == (1 if kind == "cuda" else 0)
    assert torch_stub.mps.synchronize.call_count == (1 if kind == "mps" else 0)


def test_loader_passes_device_dtype_and_revision(torch_stub, monkeypatch):
    model = Mock()
    factory = SimpleNamespace(from_pretrained=Mock(return_value=(model, {})))
    config = SimpleNamespace(model_type="other")
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=object())),
        AutoModelForCausalLM=factory,
        __version__="test",
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    torch_stub.device = lambda name: name
    torch_stub.float16 = "float16"
    torch_stub.__version__ = "test"
    revision = "a" * 40

    _, _, metadata = load_causal_model("test/model", revision, "mps", "float16")

    assert factory.from_pretrained.call_args.kwargs["device_map"] == {"": "mps"}
    assert factory.from_pretrained.call_args.kwargs["dtype"] == "float16"
    assert metadata["device"] == "mps"
    assert metadata["dtype"] == "float16"

