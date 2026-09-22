"""Compatibility import for :mod:`fastjev._runtime.direct`."""

from fastjev._runtime.direct import *  # noqa: F403
from fastjev._runtime import direct as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
