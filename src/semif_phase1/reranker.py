"""Compatibility import for :mod:`fastjev._runtime.reranker`."""

from fastjev._runtime.reranker import *  # noqa: F403
from fastjev._runtime import reranker as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
