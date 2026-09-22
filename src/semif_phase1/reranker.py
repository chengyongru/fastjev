"""Compatibility import for :mod:`fastjev.runtime.reranker`."""

from fastjev.runtime.reranker import *  # noqa: F403
from fastjev.runtime import reranker as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
