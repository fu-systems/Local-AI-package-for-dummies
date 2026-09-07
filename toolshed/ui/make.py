"""Easy mode: type what you want, press the button, get the thing.

The brief asked for something like Automatic1111 -- a prompt box and a button --
rather than a node graph, because a beginner who has just installed this cannot
wire up a sampler and should not have to. ComfyUI is still there, with the same
workflows, for when they want it.

What happens when the button is pressed:

1. ask the running engine what nodes it has (``/object_info``);
2. convert the same workflow we injected into the sidebar into API format;
3. write the prompt, size and a fresh seed into it;
4. queue it, and follow the websocket for progress;
5. fetch what came out and show it.

The engine does the work. This screen only decides what to ask for, which is
why it can drive picture, video, music and 3D workflows without knowing
anything about any of them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from toolshed import resources
from toolshed.catalog.presets import VideoPreset, default_preset, presets_for
from toolshed.easy import knobs as knobs_module
from toolshed.easy.convert import ConversionError, specs_from_object_info, to_api
from toolshed.easy.knobs import Settings, analyse
from toolshed.exec.comfy_api import ComfyClient, ComfyError, Output, Progress
from toolshed.exec.inject import PACK_FOLDERS

# What each pack is for, in the words the button should use.
VERB = {
    "image.zimage": "Make a picture",
    "image.sdxl": "Make a picture",
    "image.qwen_edit": "Change the picture",
    "video.wan22": "Make a video",
    "audio.acestep": "Make music",
    "model3d.trellis2": "Make a 3D model",
}

PLACEHOLDER = {
    "image.zimage": "a lighthouse in a storm, painted in thick oils",
    "image.sdxl": "a lighthouse in a storm, painted in thick oils",
    "image.qwen_edit": "make it night time, with the windows lit",
    "video.wan22": "a paper boat drifting down a rain-filled gutter",
    "audio.acestep": "slow piano, warm tape hiss, rain outside",
    "model3d.trellis2": "",
}


# Every "not ready yet" line, so _sync can recognise its own messages and
# clear them without wiping a result or an error.
NOT_READY_MESSAGES = frozenset({
    "Start ComfyUI first — it does the actual work.",
    "Nothing installed yet that easy mode can drive.",
})


def starter_picture() -> Path | None:
    """The picture shipped with Toolshed, for workflows that need one to begin.

    Easy mode has to work on the first press of the button. A workflow that
    starts from a photo used to leave the button unpressable until one was
    chosen, which is the opposite of the point -- the whole idea is that
    everything already has an answer and you change the ones you care about.
    """
    path = resources.resource_path("assets", "starter-photo.png")
    return path if path.is_file() else None


@dataclass(frozen=True)
class Recipe:
    """A pack, and the workflow easy mode runs for it."""

    pack_id: str
    name: str
    workflow: Path

    @property
    def verb(self) -> str:
        return VERB.get(self.pack_id, "Make it")


def recipes_for(pack_ids: list[str] | tuple[str, ...]) -> list[Recipe]:
    """The packs easy mode can drive, out of the ones installed.

    A pack whose workflow is missing from the bundle is left out rather than
    offered and then failing on click.
    """
    from toolshed.catalog.packs import load_packs

    names = {pack.id: pack.name for pack in load_packs()}
    found: list[Recipe] = []
    for pack_id in pack_ids:
        for rel in PACK_FOLDERS.get(pack_id, ()):
            path = resources.resource_path("workflows", rel)
            if path.is_file():
                found.append(Recipe(pack_id, names.get(pack_id, pack_id), path))
                break
    return found


class InspectWorker(QtCore.QThread):
    """Convert the workflow and work out what it exposes, off the GUI thread.

    Settings have to exist before the button is pressed, not after: the whole
    point is choosing what to make. Working them out needs /object_info from
    the running engine, so it cannot happen at start-up either -- it happens
    when a workflow is picked and the engine is up.
    """

    inspected = QtCore.Signal(object)         # Knobs
    failed = QtCore.Signal(str)

    def __init__(self, client: ComfyClient, workflow: Path, specs_cache: dict) -> None:
        super().__init__()
        self.client = client
        self.workflow = workflow
        self.specs_cache = specs_cache

    def run(self) -> None:      # noqa: D102 -- QThread entry point
        try:
            import json

            if "specs" not in self.specs_cache:
                self.specs_cache["specs"] = specs_from_object_info(self.client.object_info())
            specs = self.specs_cache["specs"]
            workflow = json.loads(self.workflow.read_text(encoding="utf-8"))
            knobs = analyse(to_api(workflow, specs), specs)
        except (ComfyError, ConversionError) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:                      # noqa: BLE001
            self.failed.emit(f"Could not read this workflow: {exc}")
        else:
            self.inspected.emit(knobs)


class ControlRow:
    """One editable input, as a widget that knows how to read itself back.

    Built from what the engine said about the input rather than from a table
    here, so an input on a node nobody anticipated still gets a spin box with
    the right bounds or a dropdown with the real choices.
    """

    def __init__(self, control) -> None:
        self.control = control
        self.widget = self._build(control)
        if control.tooltip:
            self.widget.setToolTip(control.tooltip)

    @staticmethod
    def _build(control) -> QtWidgets.QWidget:
        if control.kind == "choice":
            box = QtWidgets.QComboBox()
            box.addItems(list(control.choices))
            if control.value is not None and str(control.value) in control.choices:
                box.setCurrentText(str(control.value))
            return box
        if control.kind == "bool":
            box = QtWidgets.QCheckBox()
            box.setChecked(bool(control.value))
            return box
        if control.kind == "int":
            box = QtWidgets.QSpinBox()
            # Qt spin boxes are 32-bit. A seed's max is 2**64, which would
            # raise on the way in, so the range is clamped to what Qt can hold.
            low = int(max(control.minimum if control.minimum is not None else -2**31, -2**31))
            high = int(min(control.maximum if control.maximum is not None else 2**31 - 1,
                           2**31 - 1))
            box.setRange(low, max(low, high))
            if control.step:
                box.setSingleStep(max(1, int(control.step)))
            box.setValue(int(control.value) if isinstance(control.value, (int, float)) else low)
            return box
        if control.kind == "float":
            box = QtWidgets.QDoubleSpinBox()
            box.setDecimals(3)
            box.setRange(float(control.minimum if control.minimum is not None else -1e6),
                         float(control.maximum if control.maximum is not None else 1e6))
            if control.step:
                box.setSingleStep(float(control.step))
            if isinstance(control.value, (int, float)):
                box.setValue(float(control.value))
            return box
        if control.kind == "text":
            box = QtWidgets.QPlainTextEdit()
            box.setMaximumHeight(70)
            box.setPlainText("" if control.value is None else str(control.value))
            return box
        box = QtWidgets.QLineEdit()
        box.setText("" if control.value is None else str(control.value))
        return box

    def reset(self) -> None:
        """Back to the value the workflow shipped with."""
        widget = self.widget
        default = self.control.default
        if isinstance(widget, QtWidgets.QComboBox):
            if default is not None and str(default) in self.control.choices:
                widget.setCurrentText(str(default))
        elif isinstance(widget, QtWidgets.QCheckBox):
            widget.setChecked(bool(default))
        elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            if isinstance(default, (int, float)):
                widget.setValue(type(widget.value())(default))
        elif isinstance(widget, QtWidgets.QPlainTextEdit):
            widget.setPlainText("" if default is None else str(default))
        else:
            widget.setText("" if default is None else str(default))

    def value(self):
        widget = self.widget
        if isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()
        if isinstance(widget, QtWidgets.QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QtWidgets.QSpinBox):
            return widget.value()
        if isinstance(widget, QtWidgets.QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QtWidgets.QPlainTextEdit):
            return widget.toPlainText()
        return widget.text()


class GenerateWorker(QtCore.QThread):
    """One generation, off the GUI thread."""

    progressed = QtCore.Signal(object)        # Progress
    analysed = QtCore.Signal(int, int)        # the workflow's own width, height
    note = QtCore.Signal(str)
    produced = QtCore.Signal(object)          # list[Output]
    failed = QtCore.Signal(str, str)          # message, detail

    def __init__(self, client: ComfyClient, workflow: Path, settings: Settings,
                 specs_cache: dict, picture: Path | None = None) -> None:
        super().__init__()
        self.client = client
        self.workflow = workflow
        self.settings = settings
        self.specs_cache = specs_cache
        self.picture = picture
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:      # noqa: D102 -- QThread entry point
        try:
            import json

            if "specs" not in self.specs_cache:
                self.note.emit("Asking ComfyUI what it can do…")
                self.specs_cache["specs"] = specs_from_object_info(self.client.object_info())
            specs = self.specs_cache["specs"]

            workflow = json.loads(self.workflow.read_text(encoding="utf-8"))
            prompt = to_api(workflow, specs)
            knobs = analyse(prompt, specs)
            size = knobs.current_size
            if size:
                self.analysed.emit(*size)

            settings = self.settings
            if self.picture is not None:
                # Uploaded here, not when it was chosen: the engine has to be
                # running to receive it, and this is the first moment we know
                # it is.
                self.note.emit(f"Sending {self.picture.name}…")
                settings = replace(settings,
                                   image=self.client.upload_image(self.picture))

            graph = knobs_module.apply(prompt, knobs, settings)

            # Video decodes every frame to full-resolution pixels in one go,
            # which is where a job that sampled perfectly well runs the card
            # out of memory -- minutes in, with nothing to show. Tiling that
            # step is what makes the sizes on offer safe to offer.
            if knobs.is_video:
                graph = knobs_module.use_tiled_decode(graph, specs)

            self.note.emit("Queued.")
            outputs = self.client.run(
                graph,
                on_progress=self.progressed.emit,
                should_cancel=lambda: self._stop,
                titles=knobs_module.titles(graph),
            )
        except ComfyError as exc:
            # A stop is reported too, or the screen stays on "Stopping…"
            # with the button greyed out, waiting for a result that is never
            # coming.
            if exc.reason_key == "cancelled":
                self.failed.emit("Stopped.", "")
            else:
                self.failed.emit(str(exc), exc.detail)
        except ConversionError as exc:
            self.failed.emit(f"This workflow cannot be run from here: {exc}", "")
        except Exception as exc:                      # noqa: BLE001
            self.failed.emit(f"Something went wrong: {exc}", "")
        else:
            self.produced.emit(outputs)


class MakePage(QtWidgets.QWidget):
    """The prompt box and the button."""

    def __init__(self, root: Path, vram_gb: float | None = None) -> None:
        super().__init__()
        self.root = root
        # What the card has, so video is offered at a size it can finish. None
        # means detection failed, and presets_for deliberately reads that as
        # "assume the smallest" rather than "assume the best".
        self.vram_gb = vram_gb
        self.video_presets: tuple[VideoPreset, ...] = presets_for(vram_gb)
        self._is_video = False
        self.client: ComfyClient | None = None
        self.worker: GenerateWorker | None = None
        self.recipes: list[Recipe] = []
        self._specs_cache: dict = {}
        self._last: list[Output] = []
        self.knobs = None
        self.rows: list[ControlRow] = []
        self.picture: Path | None = None
        self.inspector: InspectWorker | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 12)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("Make something")
        font = heading.font()
        font.setPointSize(font.pointSize() + 5)
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)

        picker = QtWidgets.QHBoxLayout()
        picker.addWidget(QtWidgets.QLabel("What do you want to make?"))
        self.what = QtWidgets.QComboBox()
        self.what.currentIndexChanged.connect(self._on_recipe_changed)
        picker.addWidget(self.what, 1)
        layout.addLayout(picker)

        self.prompt = QtWidgets.QPlainTextEdit()
        self.prompt.setPlaceholderText("Describe what you want…")
        self.prompt.setMaximumHeight(90)
        layout.addWidget(self.prompt)

        # How long a video should be. Up here rather than in the optional
        # panels because for video it is not an optional detail: it is the
        # difference between a job that finishes and one that runs for minutes
        # and dies at the decode. Every entry on the list fits the card, so
        # there is no wrong answer to pick and nothing to warn about.
        self.video_row = QtWidgets.QWidget()
        video_layout = QtWidgets.QHBoxLayout(self.video_row)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(QtWidgets.QLabel("Video length:"))
        self.video_size = QtWidgets.QComboBox()
        for preset in self.video_presets:
            self.video_size.addItem(preset.label, preset)
        chosen = default_preset(self.video_presets)
        if chosen is not None:
            self.video_size.setCurrentIndex(self.video_presets.index(chosen))
        self.video_size.currentIndexChanged.connect(self._apply_video_preset)
        video_layout.addWidget(self.video_size, 1)
        self.video_row.setVisible(False)
        layout.addWidget(self.video_row)

        # Starting picture. Shown only for workflows that load one -- turning a
        # photo into a 3D model, or editing a picture. Without it those packs
        # could be picked and then had nothing to act on, which is not a
        # limitation of the model but a missing box.
        self.picture_row = QtWidgets.QWidget()
        picture_layout = QtWidgets.QHBoxLayout(self.picture_row)
        picture_layout.setContentsMargins(0, 0, 0, 0)
        self.picture_label = QtWidgets.QLabel("Starting picture:")
        self.picture_name = QtWidgets.QLabel("none chosen")
        self.picture_name.setWordWrap(True)
        self.picture_button = QtWidgets.QPushButton("Choose a picture…")
        self.picture_button.clicked.connect(self.choose_picture)
        self.picture_clear = QtWidgets.QPushButton("Use the one that came with Toolshed")
        self.picture_clear.clicked.connect(self.clear_picture)
        self.picture_thumb = QtWidgets.QLabel()
        self.picture_thumb.setFixedSize(64, 64)
        self.picture_thumb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        picture_layout.addWidget(self.picture_label)
        picture_layout.addWidget(self.picture_button)
        picture_layout.addWidget(self.picture_clear)
        picture_layout.addWidget(self.picture_thumb)
        picture_layout.addWidget(self.picture_name, 1)
        self.picture_row.setVisible(False)
        layout.addWidget(self.picture_row)

        # Everything most people never need, out of the way but not hidden.
        self.more = QtWidgets.QGroupBox("Common settings — optional")
        self.more.setCheckable(True)
        self.more.setChecked(False)
        form = QtWidgets.QFormLayout(self.more)
        self.negative = QtWidgets.QLineEdit()
        self.negative.setPlaceholderText("things to avoid (optional)")
        form.addRow("Avoid", self.negative)
        size_row = QtWidgets.QHBoxLayout()
        self.width = QtWidgets.QSpinBox()
        self.height = QtWidgets.QSpinBox()
        for box in (self.width, self.height):
            # Zero means "whatever the workflow already says", shown as such.
            # Without this the boxes sit at their minimum, and merely opening
            # More settings would quietly ask for a 64x64 picture.
            box.setRange(0, 8192)
            box.setSingleStep(64)
            box.setSpecialValueText("as the workflow has it")
        size_row.addWidget(self.width)
        size_row.addWidget(QtWidgets.QLabel("×"))
        size_row.addWidget(self.height)
        size_row.addStretch(1)
        self.size_row_widget = QtWidgets.QWidget()
        self.size_row_widget.setLayout(size_row)
        form.addRow("Size", self.size_row_widget)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setEnabled(False)
        self.same_seed = QtWidgets.QCheckBox("Use the same seed every time")
        self.same_seed.toggled.connect(self.seed.setEnabled)
        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(self.seed)
        seed_row.addWidget(self.same_seed)
        seed_row.addStretch(1)
        seed_widget = QtWidgets.QWidget()
        seed_widget.setLayout(seed_row)
        form.addRow("Seed", seed_widget)
        layout.addWidget(self.more)

        # Everything else the workflow exposes, built from what the engine says
        # about each input. Collapsed by default: it is the difference between
        # easy mode being a toy and being usable, but it is not the first
        # thing a beginner should meet.
        self.all_settings = QtWidgets.QGroupBox("All settings — optional")
        self.all_settings.setCheckable(True)
        self.all_settings.setChecked(False)
        outer = QtWidgets.QVBoxLayout(self.all_settings)
        blurb = QtWidgets.QLabel(
            "Everything here already has a working value. You never have to open "
            "this — press the button and it makes something. Change anything you "
            "are curious about; Reset puts it all back.")
        blurb.setWordWrap(True)
        outer.addWidget(blurb)
        self.reset_button = QtWidgets.QPushButton("Reset to the defaults")
        self.reset_button.clicked.connect(self.reset_settings)
        reset_row = QtWidgets.QHBoxLayout()
        reset_row.addWidget(self.reset_button)
        reset_row.addStretch(1)
        outer.addLayout(reset_row)
        self.settings_area = QtWidgets.QScrollArea()
        self.settings_area.setWidgetResizable(True)
        self.settings_area.setMinimumHeight(180)
        self.settings_host = QtWidgets.QWidget()
        self.settings_form = QtWidgets.QFormLayout(self.settings_host)
        self.settings_area.setWidget(self.settings_host)
        outer.addWidget(self.settings_area)
        self.all_settings.toggled.connect(self.settings_area.setVisible)
        self.settings_area.setVisible(False)
        layout.addWidget(self.all_settings)

        self.go = QtWidgets.QPushButton("Make it")
        self.go.setDefault(True)
        self.go.clicked.connect(self.generate)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop)
        self.folder_button = QtWidgets.QPushButton("Open the folder")
        self.folder_button.clicked.connect(self.open_folder)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.go)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.folder_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setVisible(False)
        layout.addWidget(self.bar)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.preview = QtWidgets.QLabel()
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(240)
        self.preview.setText("What you make will appear here.")
        layout.addWidget(self.preview, 1)

    # -- state --------------------------------------------------------------

    def set_engine(self, client: ComfyClient | None) -> None:
        """Called when ComfyUI comes up or goes away."""
        self.client = client
        self._specs_cache.clear()
        if client is not None:
            self.inspect()
        self._sync()

    def set_packs(self, pack_ids: list[str]) -> None:
        self.recipes = recipes_for(pack_ids)
        self.what.clear()
        for recipe in self.recipes:
            self.what.addItem(recipe.name)
        self._sync()

    def _current(self) -> Recipe | None:
        index = self.what.currentIndex()
        return self.recipes[index] if 0 <= index < len(self.recipes) else None

    def _on_recipe_changed(self) -> None:
        recipe = self._current()
        if recipe is None:
            return
        self.go.setText(recipe.verb)
        self.prompt.setPlaceholderText(
            PLACEHOLDER.get(recipe.pack_id) or "Describe what you want…")
        # Back to "as the workflow has it", so this workflow's own size fills
        # the boxes in. Left alone, a picture workflow's 1024x1024 would carry
        # over to a video one and be sent as an override it never asked for.
        self.width.setValue(0)
        self.height.setValue(0)
        self.inspect()
        self._sync()

    # -- what this workflow offers -------------------------------------------

    def shutdown(self) -> None:
        """Stop any worker before this page goes away.

        Qt aborts the whole process if a QThread is destroyed while running --
        "QThread: Destroyed while thread is still running" -- so a page closed
        while it was still asking the engine about a workflow would take the
        app down with it. Waiting is bounded: neither worker blocks on anything
        without a timeout of its own.
        """
        for worker in (self.inspector, self.worker):
            if worker is None:
                continue
            if hasattr(worker, "stop"):
                worker.stop()
            if worker.isRunning():
                worker.wait(5000)
        self.inspector = None
        self.worker = None

    def inspect(self) -> None:
        """Ask the workflow what it can be told, and build controls for it."""
        recipe = self._current()
        if recipe is None or self.client is None:
            return
        # One at a time. Switching workflows quickly would otherwise leave
        # several in flight and the last to answer would win, not the last
        # chosen.
        if self.inspector is not None and self.inspector.isRunning():
            self.inspector.wait(5000)
        self.inspector = InspectWorker(self.client, recipe.workflow, self._specs_cache)
        self.inspector.inspected.connect(self._on_inspected)
        self.inspector.failed.connect(self.status.setText)
        self.inspector.start()

    def _on_inspected(self, knobs) -> None:
        self.knobs = knobs

        # A workflow with no text encoder has nothing to do with a prompt box.
        self.prompt.setVisible(knobs.takes_text)
        self._is_video = knobs.is_video and bool(self.video_presets)
        self.video_row.setVisible(self._is_video)
        self.picture_row.setVisible(knobs.takes_picture)
        if knobs.takes_picture and self.picture is None:
            starter = starter_picture()
            if starter is not None:
                self.set_picture(starter, is_starter=True)
        self.negative.setEnabled(bool(knobs.negative))
        self.size_row_widget.setEnabled(knobs.has_size)
        size = knobs.current_size
        if size:
            self._on_analysed(*size)
        # After _on_analysed, which fills the size boxes with the workflow's
        # own size. For video that is the template's 1280x704, which is exactly
        # the size we are here to stop being asked for on a card that cannot
        # finish it.
        self._apply_video_preset()

        while self.settings_form.rowCount():
            self.settings_form.removeRow(0)
        self.rows = []
        last_title = None
        for control in knobs.advanced:
            if control.node_title != last_title:
                heading = QtWidgets.QLabel(f"<b>{control.node_title}</b>")
                self.settings_form.addRow(heading)
                last_title = control.node_title
            row = ControlRow(control)
            self.rows.append(row)
            self.settings_form.addRow(control.label, row.widget)

        self.all_settings.setTitle(f"All settings — optional ({len(self.rows)})")
        self.all_settings.setVisible(bool(self.rows))
        self._sync()

    def reset_settings(self) -> None:
        """Put every control back to what the workflow shipped with.

        The safety net that makes the panel safe to explore: nothing in here
        can be got so wrong that the button stops working.
        """
        for row in self.rows:
            row.reset()
        self.width.setValue(0)
        self.height.setValue(0)
        self.negative.clear()
        self.same_seed.setChecked(False)
        if self.knobs is not None:
            size = self.knobs.current_size
            if size:
                self._on_analysed(*size)
        # Reset means "back to what works", which for video is the preset for
        # this card -- not the template's size, which is what the card could
        # not finish in the first place.
        self._apply_video_preset()

    # -- the starting picture -------------------------------------------------

    def choose_picture(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose a picture", str(Path.home()),
            "Pictures (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)")
        if chosen:
            self.set_picture(Path(chosen))

    def set_picture(self, path: Path, *, is_starter: bool = False) -> None:
        self.picture = path
        self.picture_name.setText(
            "the one that came with Toolshed — swap it for your own"
            if is_starter else path.name)
        self.picture_clear.setVisible(not is_starter)
        thumb = QtGui.QPixmap(str(path))
        if not thumb.isNull():
            self.picture_thumb.setPixmap(thumb.scaled(
                self.picture_thumb.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))
        self._sync()

    def clear_picture(self) -> None:
        """Back to the picture that came with Toolshed, never to nothing."""
        starter = starter_picture()
        if starter is not None:
            self.set_picture(starter, is_starter=True)
            return
        self.picture = None
        self.picture_name.setText("none chosen")
        self.picture_thumb.clear()
        self._sync()

    def _sync(self) -> None:
        busy = self.worker is not None and self.worker.isRunning()
        ready = self.client is not None and bool(self.recipes) and not busy

        self.go.setEnabled(ready)

        # One place decides the "not ready yet" line, and one place clears it.
        # Setting these in separate branches is how "Start ComfyUI first" once
        # survived ComfyUI starting, and how "Choose a starting picture" then
        # survived a picture being chosen.
        blocked = ""
        if self.client is None:
            blocked = "Start ComfyUI first — it does the actual work."
        elif not self.recipes:
            blocked = "Nothing installed yet that easy mode can drive."

        if blocked:
            self.status.setText(blocked)
        elif not busy and self.status.text() in NOT_READY_MESSAGES:
            # Cleared only when it is one of ours, so a result or an error the
            # user still wants to read is left alone.
            self.status.setText("")

    # -- doing it -----------------------------------------------------------

    def _apply_video_preset(self) -> None:
        """Write the chosen video size into the size boxes.

        So the optional panel shows what is actually going to be asked for,
        rather than the template's size while something else is sent. Typing
        over the top still wins -- until the dropdown is used again, which is
        someone choosing a size and should overrule what was there.
        """
        preset = self.current_video_preset()
        if preset is None:
            return
        self.width.setValue(preset.width)
        self.height.setValue(preset.height)

    def current_video_preset(self) -> VideoPreset | None:
        """The chosen video size, or None when this is not a video workflow.

        Keyed off what the graph said, not off ``video_row.isVisible()``: Qt
        reports a widget as not visible whenever any ancestor is unshown, so
        asking the widget would silently drop the preset whenever this page is
        not the one on top of the stack -- and then only for some callers.
        """
        if not self._is_video:
            return None
        data = self.video_size.currentData()
        return data if isinstance(data, VideoPreset) else None

    def settings(self) -> Settings:
        overrides = {row.control.key: row.value() for row in self.rows
                     if row.control.value is not None}
        # Width and height come from the boxes, as they always have. For video
        # the preset has already written itself into them, so what the optional
        # panel shows is what is actually going to be asked for -- and a number
        # typed over the top still wins, because it is simply what the box says
        # by the time this reads it.
        preset = self.current_video_preset()
        return Settings(
            prompt=self.prompt.toPlainText().strip() or None,
            negative=self.negative.text().strip() or None,
            width=self.width.value() or None,
            height=self.height.value() or None,
            length=preset.length if preset else None,
            seed=self.seed.value() if self.same_seed.isChecked() else None,
            overrides=overrides,
        )

    def generate(self) -> None:
        recipe = self._current()
        if recipe is None or self.client is None:
            return

        self.worker = GenerateWorker(self.client, recipe.workflow, self.settings(),
                                     self._specs_cache, picture=self.picture)
        self.worker.progressed.connect(self._on_progress)
        self.worker.analysed.connect(self._on_analysed)
        self.worker.note.connect(self.status.setText)
        self.worker.produced.connect(self._on_produced)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._sync)

        self.go.setEnabled(False)
        self.stop_button.setVisible(True)
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.status.setText("Starting…")
        self.worker.start()

    def _on_analysed(self, width: int, height: int) -> None:
        """Fill the size boxes in with what the workflow actually uses.

        Only while they still say "as the workflow has it": once someone has
        chosen a size, replacing it would undo their choice mid-run.
        """
        if self.width.value() == 0:
            self.width.setValue(width)
        if self.height.value() == 0:
            self.height.setValue(height)

    def stop(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status.setText("Stopping…")

    def _on_progress(self, progress: Progress) -> None:
        self.bar.setValue(int(progress.fraction * 1000))
        where = f" — {progress.node_title}" if progress.node_title else ""
        if progress.maximum:
            self.status.setText(f"Step {progress.value} of {progress.maximum}{where}")
        else:
            self.status.setText(f"Working{where}")

    def _on_produced(self, outputs: list[Output]) -> None:
        self.bar.setVisible(False)
        self.stop_button.setVisible(False)
        self._last = outputs
        self._sync()

        if not outputs:
            self.status.setText("It finished, but produced no files.")
            return

        pictures = [o for o in outputs if o.is_picture]
        if pictures and self.client:
            self._show_picture(pictures[0])
            self.status.setText(f"Done. Saved to {self.root / 'output'}.")
        else:
            kinds = ", ".join(sorted({o.kind for o in outputs}))
            self.preview.setText(
                f"Made {len(outputs)} file(s) — {kinds}.\nOpen the folder to play it.")
            self.status.setText(f"Done. Saved to {self.root / 'output'}.")

    def _show_picture(self, item: Output) -> None:
        """Fetch the picture over the API rather than guessing where it landed.

        The engine knows the filename and subfolder it chose; reconstructing
        that path ourselves would break the moment a workflow used a different
        filename prefix.
        """
        assert self.client is not None
        try:
            import httpx

            data = httpx.get(self.client.view_url(item), timeout=60).content
        except Exception:                              # noqa: BLE001
            self.preview.setText("Made a picture. Open the folder to see it.")
            return

        image = QtGui.QPixmap()
        if not image.loadFromData(data):
            self.preview.setText("Made a picture. Open the folder to see it.")
            return
        self.preview.setPixmap(image.scaled(
            self.preview.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))

    def _on_failed(self, message: str, detail: str) -> None:
        self.bar.setVisible(False)
        self.stop_button.setVisible(False)
        self._sync()
        self.status.setText(message)
        if detail:
            self.status.setToolTip(detail)

    def open_folder(self) -> None:
        folder = self.root / "output"
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))


def a_random_seed() -> int:
    return random.randrange(0, knobs_module.SEED_MAX)
