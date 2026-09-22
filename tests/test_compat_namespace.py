"""Compatibility and dependency-direction checks for the inherited namespace."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap

import fastjev
from fastjev.cli import main as cli_main
from fastjev.compat.wire import SystemOneService
from fastjev.http import create_app
from fastjev._runtime.core import load_causal_model
from fastjev._runtime.reranker import _answer_ids
from fastjev.server import main as server_main
from semif_phase1 import __version__ as legacy_version
from semif_phase1.api import create_app as legacy_create_app
from semif_phase1.cli import main as legacy_cli_main
from semif_phase1.core import load_causal_model as legacy_load_causal_model
from semif_phase1.reranker import _answer_ids as legacy_answer_ids
from semif_phase1.server import main as legacy_server_main
from semif_phase1.system_one import SystemOneService as LegacySystemOneService


ROOT = Path(__file__).resolve().parents[1]


def test_inherited_namespace_forwards_to_fastjev():
    assert legacy_version == fastjev.__version__
    assert legacy_create_app is create_app
    assert legacy_cli_main is cli_main
    assert legacy_server_main is server_main
    assert legacy_load_causal_model is load_causal_model
    assert legacy_answer_ids is _answer_ids
    assert LegacySystemOneService is SystemOneService


def test_low_level_runtime_uses_a_private_namespace():
    assert importlib.util.find_spec("fastjev.runtime") is None
    assert importlib.util.find_spec("fastjev._runtime") is not None


def test_fastjev_imports_without_the_inherited_namespace():
    script = textwrap.dedent("""
        import importlib.abc
        import sys

        class RejectInheritedNamespace(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "semif_phase1" or fullname.startswith("semif_phase1."):
                    raise RuntimeError(f"unexpected inherited import: {fullname}")
                return None

        sys.meta_path.insert(0, RejectInheritedNamespace())
        import fastjev
        import fastjev.cli
        import fastjev.http
        import fastjev.server
        import fastjev._runtime.llama_cpp
        assert not any(name == "semif_phase1" or name.startswith("semif_phase1.") for name in sys.modules)
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
