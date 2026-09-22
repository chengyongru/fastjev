"""Compatibility import for :mod:`fastjev.runtime.serial`."""

from fastjev.runtime.serial import *  # noqa: F403
from fastjev.runtime import serial as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
