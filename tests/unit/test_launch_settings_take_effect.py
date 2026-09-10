"""The memory settings must never look applied when they are not.

Both "Use less graphics memory" and "Stream model layers into the card" are
read once, when the engine starts. That is fine in itself -- they are process
flags and there is nowhere else to read them -- but it means the checkbox and
the running engine can disagree, and the screen used to say nothing at all
when they did.

The path that made it worse: once ComfyUI is up, the button reads "Open ComfyUI
again", and start_engine returned early to reopen the browser. Every line that
writes a setting sat *after* that return. So ticking a box on a running engine
wrote nothing, applied nothing, and reported nothing -- the same ComfyUI came
back, the next video died the same way, and the reasonable conclusion was that
the option does not work.

These tests are about that gap between what the screen shows and what the
engine is doing, so they check the settings survive, the person is told, and
the running job is not killed to make it true.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 is not installed in this environment")

from PySide6 import QtWidgets  # noqa: E402

from toolshed.exec.engine import read_layer_streaming, read_low_memory  # noqa: E402
from toolshed.ui.launch import LaunchPage  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(["tests"])


class FakeEngine:
    """Up and answering, which is the only thing start_engine asks it."""

    url = "http://127.0.0.1:8188"

    def is_running(self) -> bool:
        return True


@pytest.fixture
def page(qapp, tmp_path, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("toolshed.ui.launch.open_in_browser",
                        lambda url: opened.append(url) or True)
    page = LaunchPage(tmp_path)
    page.opened = opened
    return page


def running(page) -> None:
    """Put the page in the state it is in after a successful start."""
    page.engine = FakeEngine()
    page._running_with = page.memory_settings()
    page._refresh_restart_note()


def note_shown(page) -> bool:
    """Whether the note is on, as far as the widget is concerned.

    Deliberately not the isVisible() predicate: Qt reports every widget in an
    unshown window as not visible, so that call is False here whatever the code
    did, and a "no note" assertion would pass for the wrong reason. isHidden()
    answers what the widget was actually told.
    """
    return not page.restart_note.isHidden()


class TestNothingRunning:
    def test_no_restart_note_before_anything_starts(self, page):
        page.layer_streaming.setChecked(True)
        assert not note_shown(page)
        assert not page.settings_are_stale()


class TestTickedWhileRunning:
    def test_the_choice_is_not_thrown_away(self, page, tmp_path):
        """The regression. This wrote nothing at all: every write sat after an
        early return, so the setting did not even survive to the next start."""
        running(page)
        page.layer_streaming.setChecked(True)
        page.start_engine()
        assert read_layer_streaming(tmp_path) is True, (
            "ticking the box on a running engine must at least apply next time")

    def test_the_person_is_told_this_run_is_unchanged(self, page):
        """Silence here is the whole bug: the same ComfyUI comes back and
        nothing says the setting is not in force."""
        running(page)
        page.layer_streaming.setChecked(True)
        page.start_engine()
        said = page.status.text().lower()
        assert "already running" in said
        assert "stop comfyui" in said

    def test_the_note_appears_the_moment_the_box_changes(self, page):
        """Before pressing anything -- the point is to catch them at the tick,
        not after another failed video."""
        running(page)
        assert not note_shown(page)
        page.layer_streaming.setChecked(True)
        assert note_shown(page)

    def test_it_does_not_reopen_the_browser_on_a_stale_engine(self, page):
        """Reopening the tab is what made this look like it had worked."""
        running(page)
        page.layer_streaming.setChecked(True)
        page.start_engine()
        assert page.opened == []

    def test_the_running_job_is_not_killed_to_apply_it(self, page):
        """Restarting on their behalf would take a video down part-way
        through. The engine keeps running; they choose when to stop it."""
        running(page)
        page.layer_streaming.setChecked(True)
        page.start_engine()
        assert page.engine is not None and page.engine.is_running()

    def test_low_memory_is_the_same_story(self, page, tmp_path):
        running(page)
        page.low_memory.setChecked(True)
        page.start_engine()
        assert read_low_memory(tmp_path) is True
        assert note_shown(page)

    def test_putting_it_back_clears_the_note(self, page):
        """Unticking returns the boxes to what the engine is running, so there
        is nothing left to warn about."""
        running(page)
        page.layer_streaming.setChecked(True)
        assert note_shown(page)
        page.layer_streaming.setChecked(False)
        assert not note_shown(page)


class TestUnchangedWhileRunning:
    def test_pressing_open_again_still_just_reopens(self, page):
        """The ordinary case must not have become a lecture."""
        running(page)
        page.start_engine()
        assert page.opened == [FakeEngine.url]
        assert not note_shown(page)


class TestOnceItStops:
    def test_a_stopped_engine_leaves_nothing_stale(self, page):
        """With nothing running there is no old setting to disagree with, and
        a note still on screen would be telling them to restart what is
        already down."""
        running(page)
        page.layer_streaming.setChecked(True)
        assert note_shown(page)
        page.engine = None
        page.stop_engine()
        assert not note_shown(page)
        assert not page.settings_are_stale()
