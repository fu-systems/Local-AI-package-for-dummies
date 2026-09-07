"""The application window.

Qt is imported at module scope here, but this module itself is only imported
lazily from :mod:`toolshed.__main__`, so a ``--version`` or ``--diagnostics``
call never pays for Qt and never fails because a platform plugin is missing.
"""

from __future__ import annotations

import sys

from PySide6 import QtCore, QtGui, QtWidgets

from toolshed import APP_NAME, __version__
from toolshed.catalog.packs import load_packs
from toolshed.desktop import desktop_file_installed
from toolshed.exec.engine import Layout
from toolshed.hw import HardwareReport, Verdict, detect, verdict_for
from toolshed.planner import build_plan
from toolshed.ui.choose import ChoosePage
from toolshed.ui.install import InstallPage
from toolshed.ui.launch import LaunchPage
from toolshed.ui.make import MakePage
from toolshed.ui.ready import ReadyPage, default_data_root

_YES = "✓"   # check mark
_NO = "✗"    # ballot x



class VerdictPage(QtWidgets.QWidget):
    """The first thing a user sees: what this computer can and cannot do.

    Everything here is deliberately plain English. The technical detail exists,
    but it lives behind "What does this mean?" rather than leading.
    """

    def __init__(self, report: HardwareReport, verdict: Verdict) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        headline = QtWidgets.QLabel(verdict.headline)
        headline.setWordWrap(True)
        font = headline.font()
        font.setPointSize(font.pointSize() + 5)
        font.setBold(True)
        headline.setFont(font)
        layout.addWidget(headline)

        for line in verdict.detail:
            label = QtWidgets.QLabel(line)
            label.setWordWrap(True)
            layout.addWidget(label)

        if verdict.modalities:
            layout.addSpacing(6)
            box = QtWidgets.QGroupBox("What you can make")
            box_layout = QtWidgets.QVBoxLayout(box)
            for name, ok, note in verdict.modalities:
                row = QtWidgets.QLabel(f"{_YES if ok else _NO}  <b>{name}</b> — {note}")
                row.setWordWrap(True)
                row.setEnabled(ok)
                box_layout.addWidget(row)
            layout.addWidget(box)

        details = QtWidgets.QTextEdit()
        details.setReadOnly(True)
        details.setPlainText(_technical_summary(report))
        details.setVisible(False)
        details.setMaximumHeight(160)

        toggle = QtWidgets.QPushButton("What does this mean?")
        toggle.setCheckable(True)
        toggle.toggled.connect(details.setVisible)
        toggle.toggled.connect(
            lambda on: toggle.setText("Hide the details" if on else "What does this mean?")
        )

        row = QtWidgets.QHBoxLayout()
        row.addWidget(toggle)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(details)
        layout.addStretch(1)


def _technical_summary(report: HardwareReport) -> str:
    lines = [f"Operating system: {report.os}"]
    if not report.gpus:
        lines.append("Graphics cards: none detected")
    for gpu in report.gpus:
        bits = [f"  {gpu.vendor}: {gpu.name or '(unnamed)'}"]
        if gpu.vram_gb is not None:
            bits.append(f"{gpu.vram_gb:g} GB")
        if gpu.gfx:
            bits.append(gpu.gfx)
        if gpu.driver_version:
            bits.append(f"driver {gpu.driver_version}")
        if not gpu.discrete:
            bits.append("(integrated)")
        lines.append(" | ".join(bits))
    lines.extend(f"note: {note}" for note in report.notes)
    return "\n".join(lines)


# Reasons where pressing Try again would produce the identical failure with
# nothing the user could have changed in between. Offering a button that is
# certain to fail is worse than not offering one.
#
# Deliberately NOT here: terms_required and auth_required (accept the licence
# on the web, then retry works), torch_unusable (a driver or a group membership
# fixed outside the app, then retry works), and disk_full (free some space).
RETRY_IS_POINTLESS = {"running_as_root", "not_found"}


class MainWindow(QtWidgets.QMainWindow):
    """A three-step wizard.

    The first build stopped dead on the hardware verdict: it told you what your
    machine could do and then offered no way forward at all. The pages are
    stacked behind one navigation bar so there is always an obvious next step,
    and always a way back.
    """

    def __init__(self, report: HardwareReport, verdict: Verdict) -> None:
        super().__init__()
        self.report = report
        self.verdict = verdict
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(760, 620)

        self.pages = QtWidgets.QStackedWidget()
        self.verdict_page = VerdictPage(report, verdict)
        self.pages.addWidget(self.verdict_page)

        # Unsupported hardware gets one honest screen and no further steps.
        # Walking someone through choosing packs we cannot install would be a
        # lie told in three parts.
        self.choose_page: ChoosePage | None = None
        self.ready_page: ReadyPage | None = None
        self.install_page: InstallPage | None = None
        self.launch_page: LaunchPage | None = None
        self.make_page: MakePage | None = None
        if verdict.supported:
            self.choose_page = ChoosePage(load_packs(), report)
            self.choose_page.selection_changed.connect(self._sync_nav)
            self.ready_page = ReadyPage()
            self.install_page = InstallPage()
            self.install_page.done.connect(self._on_install_done)
            self.install_page.failed.connect(self._on_install_failed)
            self.install_page.stopped.connect(self._on_install_stopped)
            self.launch_page = LaunchPage(default_data_root())
            self.launch_page.want_more_packs.connect(self._go_choose_packs)
            gpu = report.primary
            self.make_page = MakePage(default_data_root(),
                                      vram_gb=gpu.vram_gb if gpu else None)
            self.launch_page.want_easy_mode.connect(self._go_easy_mode)
            self.launch_page.engine_ready.connect(self._on_engine_ready)
            self.launch_page.engine_stopped.connect(
                lambda: self.make_page.set_engine(None))
            self.pages.addWidget(self.choose_page)
            self.pages.addWidget(self.ready_page)
            self.pages.addWidget(self.install_page)
            self.pages.addWidget(self.launch_page)
            self.pages.addWidget(self.make_page)

        self.back_button = QtWidgets.QPushButton("Back")
        self.back_button.clicked.connect(self._go_back)
        self.next_button = QtWidgets.QPushButton()
        self.next_button.setDefault(True)
        # Connected once. Unsupported hardware has nowhere to go next, so the
        # primary action is simply to close.
        if verdict.supported:
            self.next_button.clicked.connect(self._go_next)
        else:
            self.next_button.clicked.connect(self.close)

        nav = QtWidgets.QHBoxLayout()
        nav.setContentsMargins(28, 0, 28, 16)
        nav.addWidget(self.back_button)
        nav.addStretch(1)
        nav.addWidget(self.next_button)

        central = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.pages, 1)
        column.addLayout(nav)
        self.setCentralWidget(central)

        self._installing = False
        self._finished = False
        self._failed = False

        # Someone who has already installed wants to open ComfyUI, not to be
        # walked through installing it a second time. Skip straight there.
        self._already_installed = (
            verdict.supported and not Layout(default_data_root()).missing_pieces())
        if self._already_installed and self.launch_page:
            self.pages.setCurrentWidget(self.launch_page)

        self.statusBar().showMessage(
            "Nothing you make is sent anywhere." if verdict.supported
            else "Nothing has been downloaded or installed."
        )
        self._sync_nav()

    # -- navigation ---------------------------------------------------------

    def _go_next(self) -> None:
        # The primary button changes job as the wizard progresses: Continue,
        # then Set it up, then Stop while work is happening, then Close.
        if self._finished:
            self.close()
            return
        if self._failed:
            self._retry_install()
            return
        if self._installing:
            assert self.install_page
            self.install_page.stop()
            return

        index = self.pages.currentIndex()
        if self.pages.currentWidget() is self.ready_page:
            self._start_install()
            return
        if index + 1 < self.pages.count():
            if self.pages.widget(index + 1) is self.ready_page and self.choose_page:
                self.ready_page.set_selection(self.choose_page.selected())
            self.pages.setCurrentIndex(index + 1)
            self._sync_nav()

    def _start_install(self) -> None:
        assert self.choose_page and self.install_page
        plan = build_plan(self.report, self.choose_page.selected(), default_data_root())
        self.pages.setCurrentWidget(self.install_page)
        self.install_page.reset()
        # No going back once bytes are landing on disk; Stop is the way out.
        self.back_button.setVisible(False)
        self.next_button.setText("Stop")
        self.next_button.setEnabled(True)
        self._installing = True
        self.install_page.start(plan)

    def _on_install_done(self) -> None:
        """Hand the user a working ComfyUI, not a sentence about one.

        This used to set the button to Close and tell them to "open ComfyUI"
        -- a program they had just installed to a folder they did not choose,
        with no shortcut and no address. That is where "it installed but then
        nothing" came from.
        """
        self._installing = False
        self._finished = True
        self.launch_page.refresh()
        self.pages.setCurrentWidget(self.launch_page)
        self.back_button.setVisible(False)
        self.next_button.setText("Close")

    def _on_install_failed(self, message: str, reason_key: str) -> None:
        """A failure is a place to carry on from, not a dead end.

        Nothing is lost when a step fails: part-downloaded files survive, the
        workspace and the engine are kept, and every step checks what is already
        there before doing it again. So the primary offer is Try again, and it
        genuinely resumes rather than starting over.

        The one exception is a problem that trying again cannot fix -- sudo,
        or a model whose terms have to be accepted on the web first. Repeating
        the same failure on demand is not an offer, it is a loop.
        """
        self._installing = False
        self._failed = True
        self.install_page.heading.setText("That did not work")
        self.install_page.current.setText(message)

        if reason_key in RETRY_IS_POINTLESS:
            self.next_button.setText("Close")
            self._finished = True
            return

        self.install_page.hint.setText(
            "Nothing you have already downloaded is lost. Trying again picks up "
            "where this left off.")
        self.install_page.hint.setVisible(True)
        self.next_button.setText("Try again")
        self.back_button.setText("Close")
        self.back_button.setVisible(True)

    def _on_install_stopped(self) -> None:
        """Stop pressed: offer to carry on, or to leave. Never a dead end."""
        self._installing = False
        self._failed = True
        self.install_page.heading.setText("Stopped")
        self.install_page.current.setText("Setup was stopped before it finished.")
        self.install_page.hint.setText(
            "Nothing you have already downloaded is lost. Trying again picks up "
            "where this left off.")
        self.install_page.hint.setVisible(True)
        self.next_button.setText("Try again")
        self.back_button.setText("Close")
        self.back_button.setVisible(True)

    def _retry_install(self) -> None:
        self._failed = False
        self.install_page.hint.setVisible(False)
        self.back_button.setText("Back")
        self._start_install()

    def _on_engine_ready(self, url: str) -> None:
        """Easy mode can only work once something is there to do the work."""
        from toolshed.exec.comfy_api import ComfyClient
        from toolshed.exec.manifest import Manifest

        installed = Manifest.load(default_data_root()).packs
        self.make_page.set_packs(list(installed))
        self.make_page.set_engine(ComfyClient(base_url=url))

    def _go_easy_mode(self) -> None:
        self.pages.setCurrentWidget(self.make_page)
        self._sync_nav()

    def _go_choose_packs(self) -> None:
        """From the launch screen back into the wizard, to add a pack."""
        self._finished = False
        self.pages.setCurrentWidget(self.choose_page)
        self._sync_nav()

    def _go_back(self) -> None:
        if self._failed:
            self.close()
            return
        if self.pages.currentWidget() is self.make_page:
            self.pages.setCurrentWidget(self.launch_page)
            self._sync_nav()
            return
        if self.pages.currentIndex() > 0:
            self.pages.setCurrentIndex(self.pages.currentIndex() - 1)
            self._sync_nav()

    def closeEvent(self, event) -> None:      # noqa: N802 -- Qt naming
        """Never leave a ComfyUI running after the window that started it.

        It holds the graphics card and the port, and a user who closed Toolshed
        has no way left to stop it short of the task manager.
        """
        if self.make_page:
            self.make_page.shutdown()
        if self.install_page:
            # Same rule for the installer: a worker outliving the window
            # aborts the process, and an abort mid-download is the one way to
            # lose the .part the cancel path takes care to keep.
            self.install_page.shutdown()
        if self.launch_page and self.launch_page.is_running:
            self.launch_page.stop_engine()
        super().closeEvent(event)

    def _sync_nav(self) -> None:
        index = self.pages.currentIndex()
        self.back_button.setVisible(index > 0)

        if not self.verdict.supported:
            self.next_button.setText("Close")
            self.next_button.setEnabled(True)
            return

        if self._failed:
            self.next_button.setText("Try again")
            self.next_button.setEnabled(True)
            return
        if self._installing:
            self.next_button.setText("Stop")
            self.next_button.setEnabled(True)
            return
        if self._finished:
            self.next_button.setText("Close")
            self.next_button.setEnabled(True)
            return
        if self.pages.currentWidget() is self.make_page:
            self.back_button.setVisible(True)
            self.back_button.setText("Back")
            self.next_button.setText("Close")
            self.next_button.setEnabled(True)
            self._finished = True
            return
        if self.pages.currentWidget() is self.launch_page:
            # The launch page carries its own buttons; the wizard's primary
            # action there is simply to leave. Back is hidden because the page
            # behind it is the install log, which is not a place to return to
            # -- "Set up more" on the page itself is the way back into setup.
            self.back_button.setVisible(False)
            self.next_button.setText("Close")
            self.next_button.setEnabled(True)
            self._finished = True
            return
        if self.pages.currentWidget() is self.ready_page:
            self.next_button.setText("Set it up")
            self.next_button.setEnabled(True)
        elif self.choose_page and self.pages.currentWidget() is self.choose_page:
            chosen = bool(self.choose_page.selected())
            self.next_button.setText("Continue")
            self.next_button.setEnabled(chosen)
            self.next_button.setToolTip("" if chosen else "Pick at least one thing to make.")
        else:
            self.next_button.setText("Continue")
            self.next_button.setEnabled(True)


def build_application(
    argv: list[str] | None = None,
) -> tuple[QtWidgets.QApplication, MainWindow]:
    """Construct the QApplication and main window without showing anything.

    Split out from :func:`run_gui` so ``--selftest`` can exercise exactly the
    same construction path that a real launch takes. Constructing a
    QApplication alone proves very little: plugin loading for platform themes
    and icon engines happens on first widget realisation, so the selftest needs
    a real window, not just an app object.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("fu.systems")
    # Only claim a desktop file we actually installed. Qt registers the name
    # with the XDG portal, and when no matching .desktop exists -- which is the
    # normal case for the portable tarball, before install.sh has run -- the
    # portal answers with a confusing error on stderr:
    #
    #   qt.qpa.services: Failed to register with host portal
    #   QDBusError(... "Could not register app ID: App info not found for 'toolshed'")
    #
    # Nothing is broken when that happens, but a beginner running from a
    # terminal should not be shown a DBus error for a feature they did not ask
    # for. Set the name when the file is there, stay quiet when it is not.
    if desktop_file_installed("toolshed"):
        QtGui.QGuiApplication.setDesktopFileName("toolshed")

    report = detect()
    window = MainWindow(report, verdict_for(report))
    return app, window  # type: ignore[return-value]


def run_gui() -> int:
    app, window = build_application()
    window.show()
    return app.exec()


def qt_version() -> str:
    return QtCore.qVersion()
