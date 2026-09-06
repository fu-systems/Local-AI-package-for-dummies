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
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from toolshed import resources
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


class GenerateWorker(QtCore.QThread):
    """One generation, off the GUI thread."""

    progressed = QtCore.Signal(object)        # Progress
    analysed = QtCore.Signal(int, int)        # the workflow's own width, height
    note = QtCore.Signal(str)
    produced = QtCore.Signal(object)          # list[Output]
    failed = QtCore.Signal(str, str)          # message, detail

    def __init__(self, client: ComfyClient, workflow: Path, settings: Settings,
                 specs_cache: dict) -> None:
        super().__init__()
        self.client = client
        self.workflow = workflow
        self.settings = settings
        self.specs_cache = specs_cache
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
            knobs = analyse(prompt)
            size = knobs.current_size
            if size:
                self.analysed.emit(*size)
            graph = knobs_module.apply(prompt, knobs, self.settings)

            self.note.emit("Queued.")
            outputs = self.client.run(
                graph,
                on_progress=self.progressed.emit,
                should_cancel=lambda: self._stop,
                titles=knobs_module.titles(graph),
            )
        except ComfyError as exc:
            if exc.reason_key != "cancelled":
                self.failed.emit(str(exc), exc.detail)
        except ConversionError as exc:
            self.failed.emit(f"This workflow cannot be run from here: {exc}", "")
        except Exception as exc:                      # noqa: BLE001
            self.failed.emit(f"Something went wrong: {exc}", "")
        else:
            self.produced.emit(outputs)


class MakePage(QtWidgets.QWidget):
    """The prompt box and the button."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.client: ComfyClient | None = None
        self.worker: GenerateWorker | None = None
        self.recipes: list[Recipe] = []
        self._specs_cache: dict = {}
        self._last: list[Output] = []

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

        # Everything most people never need, out of the way but not hidden.
        self.more = QtWidgets.QGroupBox("More settings")
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
        self._sync()

    def _sync(self) -> None:
        busy = self.worker is not None and self.worker.isRunning()
        ready = self.client is not None and bool(self.recipes) and not busy
        self.go.setEnabled(ready)
        if self.client is None:
            self.status.setText("Start ComfyUI first — it does the actual work.")
        elif not self.recipes:
            self.status.setText("Nothing installed yet that easy mode can drive.")
        elif not busy and self.status.text().startswith(("Start ComfyUI", "Nothing installed")):
            # Clear the "not ready" message once it stops being true, without
            # wiping a result or an error the user still wants to read.
            self.status.setText("")

    # -- doing it -----------------------------------------------------------

    def settings(self) -> Settings:
        return Settings(
            prompt=self.prompt.toPlainText().strip() or None,
            negative=self.negative.text().strip() or None,
            width=self.width.value() or None,
            height=self.height.value() or None,
            seed=self.seed.value() if self.same_seed.isChecked() else None,
        )

    def generate(self) -> None:
        recipe = self._current()
        if recipe is None or self.client is None:
            return

        self.worker = GenerateWorker(self.client, recipe.workflow, self.settings(),
                                     self._specs_cache)
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
