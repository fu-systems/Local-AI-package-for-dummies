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

from toolshed.exec.engine import (
    Engine,
    EngineError,
    Layout,
    choose_layer_streaming,
    choose_low_memory,
    choose_safeguards,
    open_in_browser,
    read_extra_flags,
    read_layer_streaming,
    read_low_memory,
    write_extra_flags,
    write_layer_streaming,
    write_low_memory,
)


class EngineWorker(QtCore.QThread):
    """Starts the engine and waits for it to answer, off the GUI thread."""

    line = QtCore.Signal(str)
    ready = QtCore.Signal(str)               # url
    failed = QtCore.Signal(str, str)         # message, detail
    died = QtCore.Signal(str)                # it exited without being asked to

    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self.engine = engine
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:      # noqa: D102 -- QThread entry point
        try:
            self.engine.start(on_line=self.line.emit, on_died=self.died.emit)
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
        # The memory settings the RUNNING engine was actually started with, or
        # None when nothing is running. Not the same as what the checkboxes
        # say: both are read once at startup, so the two drift apart the moment
        # somebody ticks a box while ComfyUI is up.
        self._running_with: tuple[bool, bool] | None = None

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

        # When something fails inside ComfyUI the traceback is in its log, not
        # in this window. Saying where turns "it broke" into a bug report.
        self.log_path = QtWidgets.QLabel(
            f"ComfyUI writes its own log to {Layout(self.root).log_file}")
        self.log_path.setWordWrap(True)
        self.log_path.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.log_path)

        # Not checkable: a checkable box looked like a switch, but the
        # options were applied regardless of it. What is typed here is used.
        self.flags_box = QtWidgets.QGroupBox("Extra ComfyUI options")
        flags_layout = QtWidgets.QVBoxLayout(self.flags_box)
        self.flags = QtWidgets.QLineEdit(" ".join(read_extra_flags(self.root)))
        self.flags.setPlaceholderText("--fp32-vae")
        flags_layout.addWidget(QtWidgets.QLabel(
            "Passed to ComfyUI when it starts.\n"
            "If it crashes part-way through, try --disable-async-offload, then "
            "--disable-pinned-memory. Both are on by default on AMD and both move "
            "weights by direct memory access, which is what a graphics driver "
            "fault usually points at.\n"
            "Others: --fp32-vae or --cpu-vae if a model will not run on your card, "
            "--reserve-vram 2 to leave room for your desktop."))
        flags_layout.addWidget(self.flags)

        # The one memory setting worth a switch rather than a typed flag. It
        # is checkable because it really is a switch: what it turns on is a
        # fixed, verified set, and what that set is deliberately excludes the
        # things people are usually told to try -- see LOW_MEMORY_OPTIONS.
        self.low_memory = QtWidgets.QCheckBox(
            "Use less graphics memory (slower)")
        self.low_memory.setChecked(read_low_memory(self.root))
        self.low_memory.setToolTip(
            "Puts each model back into main memory as soon as it is done, runs "
            "the text encoder on the processor, and keeps nothing between runs. "
            "Everything takes longer and much more of it fits.")
        flags_layout.addWidget(self.low_memory)
        flags_layout.addWidget(QtWidgets.QLabel(
            "Turn this on if a job dies part-way through saying it ran out of "
            "memory. It does not help with a graphics driver fault, which is a "
            "different failure and already handled."))

        # The stronger, different thing: streaming one model's weights into the
        # card a block at a time, so a model bigger than the card still runs.
        self.layer_streaming = QtWidgets.QCheckBox(
            "Stream model layers into the card (much slower, lowest memory)")
        self.layer_streaming.setChecked(read_layer_streaming(self.root))
        self.layer_streaming.setToolTip(
            "Keeps almost none of the model on the graphics card, fetching each "
            "block from main memory as it is needed. A model far larger than "
            "your card can run this way.")
        flags_layout.addWidget(self.layer_streaming)
        flags_layout.addWidget(QtWidgets.QLabel(
            "This is the last resort, and it is genuinely slow: every block "
            "crosses to the card on every step, so a twenty-step picture moves "
            "the model twenty times. Use it when something will not run at all."))

        # Both boxes are read once, when the engine starts. Ticking one while
        # ComfyUI is up therefore changes nothing until it is restarted -- and
        # the button then reads "Open ComfyUI again", which reopened the browser
        # on an engine still running the old setting and said nothing about it.
        # That is how somebody turns on layer streaming, watches the same video
        # die the same way, and concludes the option does not exist.
        self.restart_note = QtWidgets.QLabel(
            "ComfyUI is still running with the previous setting. Press "
            "Stop ComfyUI, then Open ComfyUI, for this to take effect.")
        self.restart_note.setWordWrap(True)
        self.restart_note.setVisible(False)
        flags_layout.addWidget(self.restart_note)
        self.low_memory.toggled.connect(self._refresh_restart_note)
        self.layer_streaming.toggled.connect(self._refresh_restart_note)
        layout.addWidget(self.flags_box)

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

    def memory_settings(self) -> tuple[bool, bool]:
        """What the boxes say now, in the order the engine reads them."""
        return (self.low_memory.isChecked(), self.layer_streaming.isChecked())

    def settings_are_stale(self) -> bool:
        """Is a running engine using settings the user has since changed?"""
        return (self._running_with is not None
                and self._running_with != self.memory_settings())

    def _refresh_restart_note(self) -> None:
        self.restart_note.setVisible(self.settings_are_stale())

    def start_engine(self, env: dict[str, str] | None = None) -> None:
        if self.is_running and self.engine:
            # Already up. The memory settings were read when it started, so a
            # box ticked since then is not in force -- and silently reopening
            # the browser here is what made the setting look broken. Keep the
            # choice (it applies to the next start) and say plainly why this
            # run is unchanged. Deliberately not restarting on their behalf: a
            # video part-way through would be killed by it.
            if self.settings_are_stale():
                write_low_memory(self.root, self.low_memory.isChecked())
                write_layer_streaming(self.root, self.layer_streaming.isChecked())
                self._refresh_restart_note()
                self.status.setText(
                    "ComfyUI is already running, and it started before you changed "
                    "that setting, so this run is not using it. Press Stop ComfyUI "
                    "and then Open ComfyUI to apply it.")
                return
            open_in_browser(self.engine.url)
            return

        from toolshed.exec.manifest import Manifest

        typed = self.flags.text().strip()
        write_extra_flags(self.root, typed)
        write_low_memory(self.root, self.low_memory.isChecked())
        write_layer_streaming(self.root, self.layer_streaming.isChecked())
        extra = read_extra_flags(self.root)
        manifest = Manifest.load(self.root)
        if env is None:
            # The environment the installer chose for PyTorch -- the AMD
            # HSA_OVERRIDE_GFX_VERSION for cards that need it -- must reach the
            # engine too, or the card the install proved usable is not used.
            env = dict(manifest.torch_env)

        guards = choose_safeguards(Layout(self.root).engine_dir,
                                   rocm=self._on_rocm(manifest), extra=extra)
        if guards.applied:
            names = [g.flag for g in guards.applied]
            # "a and b" reads fine; "a and b and c" does not, and there are
            # three of these now.
            flags = (" and ".join(names) if len(names) < 3
                     else f"{', '.join(names[:-1])} and {names[-1]}")
            reasons = "; ".join(g.plain_english for g in guards.applied)
            self.log.appendPlainText(
                f"Starting with {flags}. That switches off {reasons}. These have "
                f"crashed the engine on AMD cards at the moment a large model is "
                f"swapped out, after the work was already done. Changing model "
                f"takes a little longer this way; generating is not affected.")
        for guard in guards.unavailable:
            # Never silent. A safeguard we meant to apply and could not is the
            # original crash coming back, and the one thing the user must not
            # have to discover by losing another hour to it.
            self.log.appendPlainText(
                f"Warning: this version of ComfyUI does not accept {guard.flag}, "
                f"so {guard.plain_english} stays switched on. That has crashed "
                f"AMD cards at the moment a large model is swapped out. If a long "
                f"job dies partway through, this is the first thing to suspect.")

        # Layer streaming first, because --novram and --lowvram are members of
        # the same argparse group: passing both stops ComfyUI starting. Its
        # chosen flags are handed to low memory mode as though the user had
        # typed them, so the existing group logic stands the weaker one down
        # rather than a second rule having to know about the first.
        engine_dir = Layout(self.root).engine_dir
        streaming = choose_layer_streaming(
            engine_dir, enabled=self.layer_streaming.isChecked(), extra=extra)
        if streaming.applied:
            self.log.appendPlainText(
                "Streaming model layers: --novram. Almost none of the model stays "
                "on the card; each block is fetched as it is needed. This is much "
                "slower and it is what lets a model bigger than the card run.")
        for option in streaming.overridden:
            self.log.appendPlainText(
                f"Layer streaming is standing aside: you have already chosen a "
                f"memory mode in Extra ComfyUI options, and {option.flag} alongside "
                f"it would stop ComfyUI starting.")
        for option in streaming.unavailable:
            self.log.appendPlainText(
                f"Warning: this version of ComfyUI does not accept {option.flag}, "
                f"so layer streaming is not available.")

        thrift = choose_low_memory(engine_dir,
                                   enabled=self.low_memory.isChecked(),
                                   extra=[*extra, *streaming.flags])
        if thrift.applied:
            self.log.appendPlainText(
                "Low memory mode: " + ", ".join(g.flag for g in thrift.applied)
                + ". That switches off " + "; ".join(g.plain_english for g in thrift.applied)
                + ". Everything will be slower and much more of it will fit.")
        for option in thrift.overridden:
            self.log.appendPlainText(
                f"Low memory mode is leaving {option.flag} alone: you have already "
                f"chosen from that group in Extra ComfyUI options, and passing two "
                f"would stop ComfyUI starting.")
        for option in thrift.unavailable:
            self.log.appendPlainText(
                f"Warning: this version of ComfyUI does not accept {option.flag}, "
                f"so low memory mode is doing less than it says.")

        self.engine = Engine(root=self.root, env=env, extra_args=extra,
                             safe_args=[*guards.flags, *streaming.flags, *thrift.flags])

        # Say the memory setting out loud. ComfyUI's own error report prints
        # the command line but not the environment, so when someone sends a
        # log there is otherwise no way to tell whether this was on -- which
        # cost a round of "that is the old build" on the one crash it exists
        # to fix.
        alloc = self.engine.environment().get("PYTORCH_CUDA_ALLOC_CONF")
        if alloc:
            self.log.appendPlainText(f"Memory: PYTORCH_CUDA_ALLOC_CONF={alloc}")
        self.worker = EngineWorker(self.engine)
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.ready.connect(self._on_ready)
        self.worker.failed.connect(self._on_failed)
        self.worker.died.connect(self._on_died)

        self.open_button.setEnabled(False)
        self.open_button.setText("Starting…")
        self.stop_button.setVisible(True)
        self.bar.setVisible(True)
        self.status.setText(
            "Starting ComfyUI. The first time takes a minute or two while it loads "
            "your graphics card and reads the models.")
        # What this engine is actually running with, fixed for its lifetime.
        self._running_with = self.memory_settings()
        self._refresh_restart_note()
        self.worker.start()

    def _on_rocm(self, manifest) -> bool:
        """Is this install driving an AMD card through ROCm?

        The installer wrote down which PyTorch index it used, and that is the
        exact question -- the safeguards are about the ROCm transfer path, not
        about which cards happen to be plugged in. An install from before that
        was recorded falls back to asking the machine.
        """
        if manifest.torch_index:
            return "rocm" in manifest.torch_index.lower()
        from toolshed.hw.detect import detect

        return any(gpu.vendor == "amd" for gpu in detect().gpus)

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

    def _forget_running_settings(self) -> None:
        """Nothing is running, so no setting can be stale against it."""
        self._running_with = None
        self._refresh_restart_note()

    def _on_died(self, message: str) -> None:
        """The engine went away on its own, after it had been running.

        Everything on this screen still claimed it was running, and anything
        waiting on it -- a browser tab, an easy-mode generation -- was waiting
        on a program that no longer existed. Say so, show its last words, and
        put the buttons back to a state that can start it again.
        """
        self.bar.setVisible(False)
        self.stop_button.setVisible(False)
        self.address.setVisible(False)
        self.easy_button.setEnabled(False)
        self.open_button.setEnabled(True)
        self.open_button.setText("Start ComfyUI again")
        self.status.setText(
            f"{message} Its last output is below, and the full log is at "
            f"{Layout(self.root).log_file}.")
        self.show_log.setChecked(True)
        self._forget_running_settings()
        self.engine_stopped.emit()

    def _on_failed(self, message: str, detail: str) -> None:
        self.bar.setVisible(False)
        self.stop_button.setVisible(False)
        self.open_button.setEnabled(True)
        self.open_button.setText("Try again")
        self.status.setText(message)
        self._forget_running_settings()
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
        self._forget_running_settings()
        self.engine_stopped.emit()

    def open_output_folder(self) -> None:
        folder = Layout(self.root).output_dir
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))
