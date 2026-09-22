"""Compatibility import for :mod:`fastjev.server`."""

from fastjev.server import *  # noqa: F403
from fastjev import server as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))


if __name__ == "__main__":
    _implementation.main()
