"""Compatibility import for :mod:`fastjev.http`."""

from fastjev.http import *  # noqa: F403
from fastjev import http as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
