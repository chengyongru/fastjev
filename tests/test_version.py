"""Release version consistency checks."""

import tomllib
from pathlib import Path

from fastjev import __version__
from semif_phase1.api import create_app


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project_version = tomllib.load(stream)["project"]["version"]

    assert project_version == "0.1.1"
    assert __version__ == project_version
    assert create_app(object()).version == project_version
