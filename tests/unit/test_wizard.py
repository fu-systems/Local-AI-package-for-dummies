"""Tests for the wizard flow.

The first build stopped dead on the hardware verdict: it reported what your
machine could do and offered no way forward at all. These tests exist so that
cannot happen again silently.

They construct real Qt widgets under the offscreen platform, so they need no
display and no GPU.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 is not installed in this environment")

from PySide6 import QtWidgets  # noqa: E402

from toolshed.hw.detect import Gpu, HardwareReport  # noqa: E402
from toolshed.hw.verdict import verdict_for  # noqa: E402
from toolshed.ui.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["tests"])
    yield app


def window_for(report: HardwareReport) -> MainWindow:
    return MainWindow(report, verdict_for(report))


AMD_20G = HardwareReport(os="linux", gpus=(Gpu(vendor="amd", vram_mb=20480, gfx="gfx1100"),))
NVIDIA_6G = HardwareReport(os="linux", gpus=(Gpu(vendor="nvidia", name="GTX 1660", vram_mb=6144),))
NO_GPU = HardwareReport(os="linux")
AMD_ON_WINDOWS = HardwareReport(os="windows", gpus=(Gpu(vendor="amd", vram_mb=20480),))


class TestThereIsAlwaysAWayForward:
    def test_supported_hardware_has_all_four_pages(self, qapp):
        assert window_for(AMD_20G).pages.count() == 4

    def test_the_first_page_offers_a_next_step(self, qapp):
        """The bug that prompted all of this: a verdict and no exit."""
        w = window_for(AMD_20G)
        assert w.next_button.text() == "Continue"
        assert w.next_button.isEnabled()

    def test_continue_advances_and_back_returns(self, qapp):
        w = window_for(AMD_20G)
        assert not w.back_button.isVisible() or w.pages.currentIndex() == 0
        w._go_next()
        assert w.pages.currentIndex() == 1
        w._go_back()
        assert w.pages.currentIndex() == 0

    def test_the_last_page_offers_to_actually_install(self, qapp):
        w = window_for(AMD_20G)
        w._go_next()
        w._go_next()
        assert w.next_button.text() == "Set it up"
        assert w.next_button.isEnabled()

    def test_there_is_an_install_page_to_go_to(self, qapp):
        w = window_for(AMD_20G)
        assert w.install_page is not None
        assert w.pages.count() == 4


class TestUnsupportedHardwareStops:
    @pytest.mark.parametrize("report", [NO_GPU, AMD_ON_WINDOWS])
    def test_no_pack_selection_is_offered(self, qapp, report):
        """Walking someone through choosing packs we cannot install would be a
        lie told in three parts."""
        w = window_for(report)
        assert w.pages.count() == 1
        assert w.choose_page is None
        assert w.next_button.text() == "Close"


class TestSelection:
    def test_continue_is_blocked_until_something_is_chosen(self, qapp):
        w = window_for(NVIDIA_6G)
        w._go_next()
        for row in w.choose_page.rows:
            if row.checkbox.isEnabled():
                row.checkbox.setChecked(False)
        assert not w.next_button.isEnabled()
        assert w.next_button.toolTip()

    def test_a_pack_the_card_cannot_run_is_never_pre_ticked(self, qapp):
        """default_checked must not override the hardware floor, or a 6 GB card
        starts a 12 GB download for a model it cannot load."""
        w = window_for(NVIDIA_6G)
        w._go_next()
        for row in w.choose_page.rows:
            if row.reason is not None:
                assert not row.checkbox.isChecked()
                assert not row.checkbox.isEnabled()

    def test_selection_reaches_the_confirmation_page(self, qapp):
        w = window_for(AMD_20G)
        w._go_next()
        chosen = w.choose_page.selected()
        assert chosen, "the default selection should not be empty on a capable card"
        w._go_next()
        text = w.ready_page.plan.toPlainText()
        for pack in chosen:
            assert pack.name in text
            assert pack.recipe in text

    def test_blocked_packs_are_excluded_from_the_selection(self, qapp):
        w = window_for(NVIDIA_6G)
        w._go_next()
        for pack in w.choose_page.selected():
            assert pack.vram_gb_min <= 6


class TestAFailedInstallIsNotADeadEnd:
    """The install died on "a virtual environment already exists" and the only
    button was Close. Every step is resumable now, so the honest primary offer
    is Try again -- and it must genuinely resume rather than start over."""

    def _failed_window(self, qapp, reason="step_failed"):
        window = window_for(AMD_20G)
        window._on_install_failed("Could not create the private workspace.", reason)
        return window

    def test_try_again_is_the_primary_button(self, qapp):
        window = self._failed_window(qapp)
        assert window.next_button.text() == "Try again"
        assert window.next_button.isEnabled()

    def test_closing_is_still_offered(self, qapp):
        window = self._failed_window(qapp)
        assert window.back_button.text() == "Close"

    def test_the_user_is_told_nothing_is_lost(self, qapp):
        window = self._failed_window(qapp)
        assert not window.install_page.hint.isHidden()
        assert "picks up where this left off" in window.install_page.hint.text()

    def test_the_failure_message_is_shown(self, qapp):
        window = self._failed_window(qapp)
        assert "private workspace" in window.install_page.current.text()

    @pytest.mark.parametrize("reason", ["running_as_root", "not_found"])
    def test_retry_is_not_offered_where_it_would_fail_identically(self, qapp, reason):
        """Offering a button that is certain to fail again is worse than not
        offering one. sudo cannot be undone from inside this process, and a
        model that has moved will not move back."""
        window = self._failed_window(qapp, reason)
        assert window.next_button.text() == "Close"
        assert window.install_page.hint.isHidden()

    def test_sync_nav_does_not_overwrite_the_offer(self, qapp):
        """_sync_nav runs on several signals and previously knew nothing about
        the failed state, so it would quietly relabel the button."""
        window = self._failed_window(qapp)
        window._sync_nav()
        assert window.next_button.text() == "Try again"

    def test_retrying_clears_the_previous_attempt_from_the_screen(self, qapp, monkeypatch):
        """Without a reset the step list gains a second copy of every row and
        the old crosses sit above the new ticks."""
        window = self._failed_window(qapp)
        started = {}
        monkeypatch.setattr(type(window.install_page), "start",
                            lambda self, plan, token=None: started.update(plan=plan))

        rows_before = len(window.install_page._rows)
        window.install_page._rows["stale"] = None
        window._go_next()

        assert started, "Try again did not start an install"
        assert "stale" not in window.install_page._rows, "the previous attempt was left on screen"
        assert not window._failed
        assert rows_before == 0
