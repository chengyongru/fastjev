"""Compatibility import for :mod:`fastjev._runtime.llama_cpp`."""

from fastjev._runtime.llama_cpp import *  # noqa: F403
from fastjev._runtime import llama_cpp as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
