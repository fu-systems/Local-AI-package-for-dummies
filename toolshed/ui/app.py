"""The application window.

Qt is imported at module scope here, but this module itself is only imported
lazily from :mod:`toolshed.__main__`, so a ``--version`` or ``--diagnostics``
call never pays for Qt and never fails because a platform plugin is missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from toolshed import APP_NAME, __version__
from toolshed.hw import HardwareReport, Verdict, detect, verdict_for

_YES = "✓"   # check mark
_NO = "✗"    # ballot x


def _desktop_file_installed(name: str) -> bool:
    """True if <name>.desktop exists in any XDG application directory.

    Follows the XDG base directory spec: $XDG_DATA_HOME (default
    ~/.local/share) then each entry of $XDG_DATA_DIRS (default
    /usr/local/share:/usr/share).
    """
    if sys.platform != "linux":
        return True  # only the XDG portal cares; other platforms are unaffected

    data_home = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    for base in [data_home, *data_dirs.split(":")]:
        if base and Path(base, "applications", f"{name}.desktop").is_file():
            return True
    return False


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

        layout.addStretch(1)

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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, report: HardwareReport, verdict: Verdict) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(720, 560)
        self.setCentralWidget(VerdictPage(report, verdict))
        self.statusBar().showMessage(
            "Nothing you make is sent anywhere." if verdict.supported
            else "Nothing has been downloaded or installed."
        )


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
    if _desktop_file_installed("toolshed"):
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
