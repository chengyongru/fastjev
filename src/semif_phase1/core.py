"""Compatibility import for :mod:`fastjev.runtime.core`."""

from fastjev.runtime.core import *  # noqa: F403
from fastjev.runtime import core as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
