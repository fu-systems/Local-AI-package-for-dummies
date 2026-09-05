"""Tests for the null-stream guard.

This is the one piece of shipped code whose entire reason to exist is a state
that never occurs during development: a PyInstaller windowed build on Windows,
where CPython sets the standard streams to None. It cannot be caught by running
the app normally, so it is tested here and again by the Windows build workflow's
windowed smoke test.
"""

from __future__ import annotations

import io
import sys

import pytest

from toolshed._streams import NullStream, install_stream_guards


@pytest.fixture
def restore_streams():
    saved = (sys.stdout, sys.stderr, sys.stdin,
             sys.__stdout__, sys.__stderr__, sys.__stdin__)
    yield
    (sys.stdout, sys.stderr, sys.stdin,
     sys.__stdout__, sys.__stderr__, sys.__stdin__) = saved


class TestNullStream:
    def test_write_reports_the_length_it_swallowed(self):
        assert NullStream().write("hello") == 5

    def test_print_to_it_does_not_raise(self):
        print("anything at all", file=NullStream())

    def test_writelines_consumes_without_raising(self):
        NullStream().writelines(["a", "b", "c"])

    def test_it_is_a_real_text_stream(self):
        """Subclassing TextIOBase, rather than duck-typing, is what makes the
        stdlib's isinstance checks and context-manager support work."""
        stream = NullStream()
        assert isinstance(stream, io.TextIOBase)
        assert stream.writable() and not stream.readable()
        assert not stream.isatty()
        with NullStream() as s:
            s.write("closes cleanly")

    def test_fileno_raises_rather_than_inventing_one(self):
        """Handing out fd 1 in a windowed process writes into whatever handle
        the OS happened to leave there. Refusing is the only safe answer."""
        with pytest.raises(io.UnsupportedOperation):
            NullStream().fileno()


class TestInstallStreamGuards:
    def test_replaces_none_streams(self, restore_streams):
        sys.stdout = None
        sys.stderr = None
        sys.stdin = None
        install_stream_guards()
        print("this would raise AttributeError without the guard")
        assert isinstance(sys.stdout, NullStream)
        assert isinstance(sys.stderr, NullStream)
        assert sys.stdin is not None

    def test_also_repairs_the_dunder_originals(self, restore_streams):
        """logging's lastResort handler and pdb reach for sys.__stderr__
        specifically, so fixing only sys.stderr leaves a second landmine."""
        sys.stdout = None
        sys.stderr = None
        sys.__stdout__ = None
        sys.__stderr__ = None
        install_stream_guards()
        assert sys.__stdout__ is not None
        assert sys.__stderr__ is not None

    def test_never_touches_real_streams(self, restore_streams):
        real_out, real_err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = real_out, real_err
        install_stream_guards()
        assert sys.stdout is real_out
        assert sys.stderr is real_err

    def test_is_idempotent(self, restore_streams):
        sys.stdout = None
        install_stream_guards()
        first = sys.stdout
        install_stream_guards()
        assert sys.stdout is first
