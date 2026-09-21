"""Distribution metadata must satisfy PyPI's dependency policy."""

import importlib.util
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_VERSION = "9.8.7"
spec = importlib.util.spec_from_file_location(
    "verify_distribution_metadata", ROOT / "scripts/verify_distribution_metadata.py"
)
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


def wheel_with_requirements(tmp_path, *requirements):
    path = tmp_path / f"fastjev-{FIXTURE_VERSION}-py3-none-any.whl"
    fields = ["Metadata-Version: 2.4", "Name: fastjev", f"Version: {FIXTURE_VERSION}"]
    fields.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"fastjev-{FIXTURE_VERSION}.dist-info/METADATA", "\n".join(fields) + "\n\n"
        )
    return path


def test_accepts_index_dependencies(tmp_path):
    path = wheel_with_requirements(
        tmp_path,
        'mlx==0.32.2; sys_platform == "darwin"',
        'vllm==0.29.0; extra == "vllm"',
    )
    metadata.validate(path)


def test_rejects_conditional_direct_dependencies(tmp_path):
    requirement = (
        'mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@revision; '
        'sys_platform == "darwin" and extra == "mlx"'
    )
    path = wheel_with_requirements(tmp_path, requirement)
    with pytest.raises(metadata.MetadataValidationError, match="mlx-lm @ git\\+"):
        metadata.validate(path)


def test_expands_distribution_globs(tmp_path):
    wheel = wheel_with_requirements(tmp_path, "mlx==0.32.2")
    assert metadata.expand_paths([tmp_path / "*"]) == [wheel]
