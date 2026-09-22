"""Compatibility import for :mod:`fastjev.runtime.shared`."""

from fastjev.runtime.shared import *  # noqa: F403
from fastjev.runtime import shared as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
