"""Release version consistency checks."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from fastjev import __version__
from fastjev.http import create_app


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project_version = tomllib.load(stream)["project"]["version"]

    assert project_version == "0.2.0"
    assert __version__ == project_version
    assert create_app(object()).version == project_version
