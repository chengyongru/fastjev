"""Compatibility import for :mod:`fastjev.compat.wire`."""

from fastjev.compat.wire import *  # noqa: F403
from fastjev.compat import wire as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
