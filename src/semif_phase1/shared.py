"""Compatibility import for :mod:`fastjev._runtime.shared`."""

from fastjev._runtime.shared import *  # noqa: F403
from fastjev._runtime import shared as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
