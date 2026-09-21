import fastjev
import pytest

from semif_phase1.direct import score
from semif_phase1.system_one import SystemOneService


def test_public_sdk_exports_stable_entrypoints():
    assert fastjev.score_direct is score
    assert fastjev.SystemOneService is SystemOneService
    assert callable(fastjev.load_causal_model)


def test_optional_http_module_exports_app_factory():
    pytest.importorskip("fastapi")

    from fastjev.http import create_app

    assert callable(create_app)
