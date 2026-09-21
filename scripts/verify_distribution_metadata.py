#!/usr/bin/env python3
"""Reject dependency metadata that PyPI will refuse to publish."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import glob
from pathlib import Path
import tarfile
import zipfile


class MetadataValidationError(ValueError):
    """Raised when a distribution contains unsupported dependency metadata."""


def _metadata_bytes(path: Path) -> bytes:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise MetadataValidationError(f"{path}: expected one wheel METADATA file")
            return archive.read(names[0])
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = [
                member for member in archive.getmembers()
                if member.isfile() and len(Path(member.name).parts) == 2
                and Path(member.name).name == "PKG-INFO"
            ]
            if len(members) != 1:
                raise MetadataValidationError(f"{path}: expected one root PKG-INFO file")
            stream = archive.extractfile(members[0])
            if stream is None:
                raise MetadataValidationError(f"{path}: could not read PKG-INFO")
            return stream.read()
    raise MetadataValidationError(f"{path}: expected a wheel or .tar.gz source distribution")


def direct_dependencies(path: Path) -> list[str]:
    metadata = BytesParser().parsebytes(_metadata_bytes(path))
    requirements = metadata.get_all("Requires-Dist", [])
    return [requirement for requirement in requirements if " @ " in requirement.split(";", 1)[0]]


def validate(path: Path) -> None:
    rejected = direct_dependencies(path)
    if rejected:
        formatted = "\n  ".join(rejected)
        raise MetadataValidationError(
            f"{path}: PyPI does not accept direct dependency references:\n  {formatted}"
        )


def expand_paths(patterns: list[Path]) -> list[Path]:
    paths = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(str(pattern))]
        paths.extend(sorted(matches) if matches else [pattern])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distributions", nargs="+", type=Path)
    args = parser.parse_args()
    for path in expand_paths(args.distributions):
        validate(path)
        print(f"{path}: publishable dependency metadata")


if __name__ == "__main__":
    main()
