"""Stdio guards for windowed (GUI) builds.

A PyInstaller ``console=False`` build on Windows starts with no console
attached, so CPython initialises ``sys.stdout``, ``sys.stderr`` and
``sys.stdin`` to ``None``. Any code that then calls ``print()``,
``traceback.print_exc()``, ``warnings.warn`` or logging's ``lastResort``
handler raises ``AttributeError: 'NoneType' object has no attribute 'write'``
-- usually from inside a third-party library we do not control. The app dies
for a reason the user cannot act on and cannot report.

This module replaces only the streams that are actually ``None``. Real streams
are never touched, so a console build, a terminal launch on Linux, and
``toolshed --version | cat`` all behave exactly as they did before.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterable


class NullStream(io.TextIOBase):
    """A writable text stream that discards everything written to it.

    Subclassing ``io.TextIOBase`` rather than writing a duck type means the
    standard library's isinstance checks, context-manager support, ``close()``,
    ``readable()`` and the TextIOBase repr all work without us reimplementing
    them.
    """

    encoding = "utf-8"
    errors = "replace"

    def writable(self) -> bool:
        return True

    def write(self, s: str, /) -> int:
        return len(s)

    def writelines(self, lines: Iterable[str], /) -> None:
        for _ in lines:
            pass

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        # There genuinely is no file descriptor. Raising UnsupportedOperation is
        # what a detached stream does and what callers that can degrade
        # gracefully (subprocess, logging) already handle. Never invent an fd:
        # handing out 1 in a windowed process writes into whatever handle the OS
        # happened to leave there.
        raise io.UnsupportedOperation("fileno")


def install_stream_guards() -> None:
    """Replace any ``None`` standard stream with a discarding stand-in.

    Idempotent, dependency-free, and safe to call as the very first statement of
    the process. It must not import anything that might itself print.
    """
    if sys.stdout is None:
        sys.stdout = NullStream()
    if sys.stderr is None:
        sys.stderr = NullStream()
    if sys.stdin is None:
        sys.stdin = io.StringIO()

    # sys.__stdout__ and friends are ALSO None under a windowed build. Logging's
    # lastResort handler, pdb and several libraries reach for the dunder
    # originals specifically, so mirror the fix rather than leaving a second
    # landmine behind the first.
    if sys.__stdout__ is None:
        sys.__stdout__ = sys.stdout  # type: ignore[misc]
    if sys.__stderr__ is None:
        sys.__stderr__ = sys.stderr  # type: ignore[misc]
    if sys.__stdin__ is None:
        sys.__stdin__ = sys.stdin  # type: ignore[misc]
