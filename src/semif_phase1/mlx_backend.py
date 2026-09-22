"""Compatibility import for :mod:`fastjev._runtime.mlx`."""

from fastjev._runtime.mlx import *  # noqa: F403
from fastjev._runtime import mlx as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
