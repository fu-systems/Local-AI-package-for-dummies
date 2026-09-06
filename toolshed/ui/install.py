"""The install screen: a named step moving, not one opaque bar.

"It froze at 70% and I have no idea what it was doing" is the single most
common complaint about every tool in this space. So each step is listed by
name, the current one shows its own progress and transfer rate, and the real
log is one click away rather than the primary view.

The work happens on a worker thread. Qt widgets may only be touched from the
GUI thread, so the runner communicates purely through a signal.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from toolshed.exec.download import Cancelled
from toolshed.exec.runner import Event, InstallFailed, Runner
from toolshed.planner.plan import InstallPlan

TICK = "✓"      # done
CROSS = "✗"     # failed
DOT = "•"       # in progress


class InstallWorker(QtCore.QThread):
    """Runs the plan off the GUI thread."""

    event = QtCore.Signal(object)
    finished_ok = QtCore.Signal()
    failed = QtCore.Signal(str, str)     # message, reason_key
    cancelled = QtCore.Signal()

    def __init__(self, plan: InstallPlan, token: str | None = None) -> None:
        super().__init__()
        self.plan = plan
        self.token = token
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:      # noqa: D102 -- QThread entry point
        try:
            Runner(self.plan,
                   on_event=self.event.emit,
                   should_cancel=lambda: self._stop,
                   hf_token=self.token).run()
        except Cancelled:
            self.cancelled.emit()
        except InstallFailed as exc:
            self.failed.emit(str(exc), exc.reason_key)
        except Exception as exc:                      # noqa: BLE001
            # Nothing may escape a worker thread silently; an unclassified
            # failure still has to reach the user as words.
            self.failed.emit(f"Something went wrong that we did not plan for: {exc}",
                             "unclassified")
        else:
            self.finished_ok.emit()


class InstallPage(QtWidgets.QWidget):
    done = QtCore.Signal()
    failed = QtCore.Signal(str, str)
    stopped = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.worker: InstallWorker | None = None
        self._rows: dict[str, QtWidgets.QLabel] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 8)
        layout.setSpacing(10)

        self.heading = QtWidgets.QLabel("Setting up")
        f = self.heading.font()
        f.setPointSize(f.pointSize() + 5)
        f.setBold(True)
        self.heading.setFont(f)
        layout.addWidget(self.heading)

        self.current = QtWidgets.QLabel("Starting…")
        self.current.setWordWrap(True)
        layout.addWidget(self.current)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        layout.addWidget(self.bar)

        # Shown only when something has gone wrong, to say what is still safe.
        self.hint = QtWidgets.QLabel()
        self.hint.setWordWrap(True)
        self.hint.setVisible(False)
        layout.addWidget(self.hint)

        self.steps_box = QtWidgets.QWidget()
        self.steps_layout = QtWidgets.QVBoxLayout(self.steps_box)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(self.steps_box)
        layout.addWidget(scroll, 1)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        # O(1) append and a hard cap: a pip install can emit thousands of lines
        # and an unbounded document eventually stalls the UI.
        self.log.setMaximumBlockCount(2000)
        self.log.setVisible(False)
        self.log.setMaximumHeight(170)

        self.show_log = QtWidgets.QPushButton("Show details")
        self.show_log.setCheckable(True)
        self.show_log.toggled.connect(self.log.setVisible)
        self.show_log.toggled.connect(
            lambda on: self.show_log.setText("Hide details" if on else "Show details"))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.show_log)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.log)

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Clear the screen for a fresh run.

        Called before every start, including a retry. Without it the step list
        would gain a second copy of every row, and the previous run's ticks and
        crosses would sit above the new ones as though they still described
        what was happening.
        """
        while (item := self.steps_layout.takeAt(0)) is not None:
            if (widget := item.widget()) is not None:
                widget.deleteLater()
        self._rows.clear()
        self.log.clear()
        self.bar.setValue(0)
        self.hint.setVisible(False)
        self.heading.setText("Setting up")
        self.current.setText("Starting…")

    def start(self, plan: InstallPlan, token: str | None = None) -> None:
        for step in plan.steps:
            label = QtWidgets.QLabel(f"   {step.title}")
            label.setEnabled(False)
            self._rows[step.id] = label
            self.steps_layout.addWidget(label)
        self.steps_layout.addStretch(1)

        self.worker = InstallWorker(plan, token)
        self.worker.event.connect(self._on_event)
        self.worker.finished_ok.connect(self.done)
        self.worker.failed.connect(self.failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()

    def stop(self) -> None:
        if self.worker and self.worker.isRunning():
            self.current.setText("Stopping… your progress is kept.")
            self.worker.stop()

    def shutdown(self, wait_ms: int = 30_000) -> bool:
        """Stop the worker and wait for it, before this page can go away.

        Qt aborts the whole process if a QThread is destroyed while it is
        still running, so closing the window mid-install must first ask the
        runner to stop and give it time to. Everything it was doing is safe to
        interrupt: downloads keep their .part, subprocesses are killed as a
        tree, and every step picks up where it left off next time.
        """
        if self.worker is None or not self.worker.isRunning():
            return True
        self.worker.stop()
        return self.worker.wait(wait_ms)

    def _on_cancelled(self) -> None:
        """Stopping is a place to carry on from, and the screen must say so.

        This used to change one label to "Stopped." and leave a Stop button
        that no longer did anything: no way to try again, no way to close.
        """
        self.current.setText("Stopped.")
        self.stopped.emit()

    def _on_event(self, event: Event) -> None:
        if event.kind == "log":
            self.log.appendPlainText(event.message)
            return

        step_id = event.step.id if event.step else ""
        label = self._rows.get(step_id)

        if event.kind == "step_started":
            self.current.setText(event.message)
            if label:
                label.setEnabled(True)
                label.setText(f"{DOT}  {event.step.title}")
        elif event.kind == "progress":
            if event.message:
                self.current.setText(event.message)
            if label and event.bytes_total:
                label.setText(f"{DOT}  {event.step.title}   "
                              f"{event.bytes_done / 1e9:.1f} / {event.bytes_total / 1e9:.1f} GB")
        elif event.kind == "step_done":
            if label:
                label.setText(f"{TICK}  {event.step.title}")
        elif event.kind == "step_failed":
            if label:
                label.setText(f"{CROSS}  {event.step.title}")
            self.log.appendPlainText(event.message)

        # Blend whole-step progress with progress inside the current step, so
        # the bar keeps moving during a long download instead of sitting still.
        steps = max(len(self._rows), 1)
        overall = event.overall + (event.fraction / steps if event.fraction else 0.0)
        self.bar.setValue(int(min(overall, 1.0) * 1000))
