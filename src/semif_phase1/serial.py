"""Compatibility import for :mod:`fastjev._runtime.serial`."""

from fastjev._runtime.serial import *  # noqa: F403
from fastjev._runtime import serial as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
