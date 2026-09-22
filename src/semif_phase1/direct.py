"""Compatibility import for :mod:`fastjev.runtime.direct`."""

from fastjev.runtime.direct import *  # noqa: F403
from fastjev.runtime import direct as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
