"""The screen that finally opens ComfyUI.

The install used to end with a sentence -- "Everything is set up. Open ComfyUI
and your workflows are waiting" -- and no way to do either. For someone who has
never run a local model, telling them to open a program we just installed
somewhere they did not choose is the same as telling them nothing.

So this page starts the engine, waits for it to genuinely answer, and opens the
browser on it. It is also where the app starts on every later run, because once
ComfyUI is installed the thing you want from Toolshed is to launch it, not to
be walked through installing it again.

Starting is slow and must not freeze the window, so the engine is supervised
from a worker thread and talks back through signals.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from toolshed.exec.engine import Engine, EngineError, Layout, open_in_browser


class EngineWorker(QtCore.QThread):
    """Starts the engine and waits for it to answer, off the GUI thread."""

    line = QtCore.Signal(str)
    ready = QtCore.Signal(str)               # url
    failed = QtCore.Signal(str, str)         # message, detail

    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self.engine = engine
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:      # noqa: D102 -- QThread entry point
        try:
            self.engine.start(on_line=self.line.emit)
            self.engine.wait_until_ready(should_cancel=lambda: self._stop)
        except EngineError as exc:
            if exc.reason_key != "cancelled":
                self.failed.emit(str(exc), exc.detail)
        except Exception as exc:                      # noqa: BLE001
            # Nothing escapes a worker thread silently.
            self.failed.emit(f"Could not start ComfyUI: {exc}", "")
        else:
            self.ready.emit(self.engine.url)


class LaunchPage(QtWidgets.QWidget):
    """Open ComfyUI, and say what is happening while it comes up."""

    want_more_packs = QtCore.Signal()
    want_easy_mode = QtCore.Signal()
    engine_ready = QtCore.Signal(str)      # url
    engine_stopped = QtCore.Signal()

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.engine: Engine | None = None
        self.worker: EngineWorker | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 8)
        layout.setSpacing(12)

        self.heading = QtWidgets.QLabel("Ready. Let's make something.")
        font = self.heading.font()
        font.setPointSize(font.pointSize() + 5)
        font.setBold(True)
        self.heading.setFont(font)
        layout.addWidget(self.heading)

        self.blurb = QtWidgets.QLabel(
            "ComfyUI opens in your web browser. Your ready-made workflows are in the "
            "Workflows sidebar on the left, in a folder called \"Toolshed\" — open one, "
            "type what you want, and press Run.")
        self.blurb.setWordWrap(True)
        layout.addWidget(self.blurb)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.bar = QtWidgets.QProgressBar()
        # Indeterminate: startup time depends on the models on disk and we
        # would rather show honest motion than an invented percentage.
        self.bar.setRange(0, 0)
        self.bar.setVisible(False)
        layout.addWidget(self.bar)

        self.address = QtWidgets.QLabel("")
        self.address.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.address.setVisible(False)
        layout.addWidget(self.address)

        row = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open ComfyUI")
        self.open_button.clicked.connect(self.start_engine)
        row.addWidget(self.open_button)

        self.stop_button = QtWidgets.QPushButton("Stop ComfyUI")
        self.stop_button.clicked.connect(self.stop_engine)
        self.stop_button.setVisible(False)
        row.addWidget(self.stop_button)

        self.folder_button = QtWidgets.QPushButton("Open my pictures folder")
        self.folder_button.clicked.connect(self.open_output_folder)
        row.addWidget(self.folder_button)

        self.easy_button = QtWidgets.QPushButton("Make something (easy mode)")
        self.easy_button.clicked.connect(self.want_easy_mode)
        self.easy_button.setEnabled(False)
        self.easy_button.setToolTip("Start ComfyUI first — it does the work.")
        row.addWidget(self.easy_button)

        self.more_button = QtWidgets.QPushButton("Set up more")
        self.more_button.clicked.connect(self.want_more_packs)
        row.addWidget(self.more_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setVisible(False)
        self.log.setMaximumHeight(200)
        self.show_log = QtWidgets.QPushButton("Show details")
        self.show_log.setCheckable(True)
        self.show_log.toggled.connect(self.log.setVisible)
        self.show_log.toggled.connect(
            lambda on: self.show_log.setText("Hide details" if on else "Show details"))
        detail_row = QtWidgets.QHBoxLayout()
        detail_row.addWidget(self.show_log)
        detail_row.addStretch(1)
        layout.addLayout(detail_row)
        layout.addWidget(self.log)
        layout.addStretch(1)

        self.refresh()

    # -- state --------------------------------------------------------------

    def refresh(self) -> None:
        """Say plainly whether there is anything to open."""
        missing = Layout(self.root).missing_pieces()
        if missing:
            self.status.setText("Not installed yet: missing " + ", ".join(missing) + ".")
            self.open_button.setEnabled(False)
        else:
            self.status.setText("")
            self.open_button.setEnabled(True)

    @property
    def is_running(self) -> bool:
        return self.engine is not None and self.engine.is_running()

    # -- actions ------------------------------------------------------------

    def start_engine(self, env: dict[str, str] | None = None) -> None:
        if self.is_running and self.engine:
            # Already up: this is now just "show me it again".
            open_in_browser(self.engine.url)
            return

        self.engine = Engine(root=self.root, env=env or {})
        self.worker = EngineWorker(self.engine)
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.ready.connect(self._on_ready)
        self.worker.failed.connect(self._on_failed)

        self.open_button.setEnabled(False)
        self.open_button.setText("Starting…")
        self.stop_button.setVisible(True)
        self.bar.setVisible(True)
        self.status.setText(
            "Starting ComfyUI. The first time takes a minute or two while it loads "
            "your graphics card and reads the models.")
        self.worker.start()

    def _on_ready(self, url: str) -> None:
        self.bar.setVisible(False)
        self.open_button.setEnabled(True)
        self.open_button.setText("Open ComfyUI again")
        self.status.setText("ComfyUI is running.")
        self.easy_button.setEnabled(True)
        self.easy_button.setToolTip("")
        self.engine_ready.emit(url)
        self.address.setText(f"If your browser did not open, go to: {url}")
        self.address.setVisible(True)
        if not open_in_browser(url):
            self.status.setText(
                "ComfyUI is running, but we could not open your browser for you.")

    def _on_failed(self, message: str, detail: str) -> None:
        self.bar.setVisible(False)
        self.stop_button.setVisible(False)
        self.open_button.setEnabled(True)
        self.open_button.setText("Try again")
        self.status.setText(message)
        if detail:
            # The engine's own last words are the only useful thing here, so
            # show them rather than making the user go hunting for a log file.
            self.log.appendPlainText(detail)
            self.show_log.setChecked(True)

    def stop_engine(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        if self.engine:
            self.engine.stop()
        self.bar.setVisible(False)
        self.stop_button.setVisible(False)
        self.address.setVisible(False)
        self.open_button.setEnabled(True)
        self.open_button.setText("Open ComfyUI")
        self.status.setText("ComfyUI is stopped.")
        self.easy_button.setEnabled(False)
        self.engine_stopped.emit()

    def open_output_folder(self) -> None:
        folder = Layout(self.root).output_dir
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))
