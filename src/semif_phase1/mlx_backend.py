"""Compatibility import for :mod:`fastjev.runtime.mlx`."""

from fastjev.runtime.mlx import *  # noqa: F403
from fastjev.runtime import mlx as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
