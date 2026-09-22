"""Compatibility import for :mod:`fastjev.cli`."""

from fastjev.cli import *  # noqa: F403
from fastjev import cli as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))


if __name__ == "__main__":
    _implementation.main()
