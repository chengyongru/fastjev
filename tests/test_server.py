from argparse import Namespace

import pytest

from fastjev.server import _parser, _validate_args


def args(**changes):
    values = {
        "backend": "torch",
        "model": "unused",
        "revision": "unused",
        "served_model": "semif-test",
        "served_model_description": "Test model",
        "served_model_release_date": "2026-09-20",
        "max_tokens": 4096,
        "host": "127.0.0.1",
        "port": 8000,
        "api_key_env": "SEMIF_TEST_API_KEY",
        "allow_unauthenticated": False,
        "mlx_bits": None,
        "mlx_cache_limit_mib": None,
    }
    values.update(changes)
    return Namespace(**values)


def test_loopback_can_run_without_authentication(monkeypatch):
    monkeypatch.delenv("SEMIF_TEST_API_KEY", raising=False)
    assert _validate_args(_parser(), args()) is None


def test_default_api_key_environment_variable_uses_fastjev_name():
    parsed = _parser().parse_args([
        "--model", "unused",
        "--revision", "unused",
        "--served-model", "fastjev-test",
        "--served-model-description", "Test model",
        "--served-model-release-date", "2026-09-20",
    ])
    assert parsed.api_key_env == "FASTJEV_API_KEY"


def test_non_loopback_requires_authentication_or_explicit_override(monkeypatch):
    monkeypatch.delenv("SEMIF_TEST_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        _validate_args(_parser(), args(host="0.0.0.0"))
    assert _validate_args(_parser(), args(host="0.0.0.0", allow_unauthenticated=True)) is None


def test_api_key_is_read_from_named_environment_variable(monkeypatch):
    monkeypatch.setenv("SEMIF_TEST_API_KEY", "test-secret")
    assert _validate_args(_parser(), args(host="0.0.0.0")) == "test-secret"


@pytest.mark.parametrize("changes,message", [
    ({"served_model": "jev-latest"}, "must not impersonate"),
    ({"served_model_description": ""}, "must not be empty"),
    ({"served_model_release_date": "September 20"}, "must be an ISO date"),
])
def test_invalid_served_model_metadata_fails_before_loading(monkeypatch, changes, message, capsys):
    monkeypatch.delenv("SEMIF_TEST_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        _validate_args(_parser(), args(**changes))
    assert message in capsys.readouterr().err
