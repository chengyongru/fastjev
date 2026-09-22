"""Package namespace contracts."""

import importlib.util


def test_low_level_runtime_uses_a_private_namespace():
    assert importlib.util.find_spec("fastjev.runtime") is None
    assert importlib.util.find_spec("fastjev._runtime") is not None
