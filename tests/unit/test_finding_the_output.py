"""Tests for telling someone where the thing they made actually is.

A 3D run reported success, showed a mask, and said "Saved to <root>/output".
The model was in <root>/output/3d, its name was never mentioned, and two of the
three "files" the run claimed to have made were previews in ComfyUI's temp
directory that it deletes. Everything the screen said was true and none of it
helped anyone find a .glb.

The shapes below are read from ComfyUI v0.34.0, not invented:

  Save3DAdvanced    -> _save_file3d_to_output writes into
                       folder_paths.get_output_directory() and reports
                       "<subfolder>/<name>" via UI.PreviewUI3DAdvanced
  Preview3DAdvanced -> writes into folder_paths.get_temp_directory() and
                       reports a bare filename through the *same* class
  MaskPreview       -> subclasses PreviewImage, so images with type "temp"

Both 3D nodes therefore arrive as {"result": [name, ...]} and only the node
class tells them apart.
"""

from __future__ import annotations

import os

import pytest

from toolshed.exec.comfy_api import Output, outputs_from_history

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# One 3D run: a saved model, two previews of it, and a mask.
HISTORY = {
    "outputs": {
        "322": {"result": ["3d/ComfyUI_00001_.glb", None, []]},
        "246": {"result": ["preview3d_advanced_9f2c.glb", None, []]},
        "303": {"images": [{"filename": "ComfyUI_temp_abcde_00001_.png",
                            "subfolder": "", "type": "temp"}],
                "animated": [False]},
    }
}

GRAPH = {
    "322": {"class_type": "Save3DAdvanced"},
    "246": {"class_type": "Preview3DAdvanced"},
    "303": {"class_type": "MaskPreview"},
}


class TestTheWalker:
    def test_the_saved_model_keeps_its_subfolder(self):
        """output/3d, not output. This is where the model went missing."""
        found = outputs_from_history(HISTORY, GRAPH)
        model = next(o for o in found if o.filename == "ComfyUI_00001_.glb")
        assert model.subfolder == "3d"
        assert model.type == "output" and model.is_saved
        assert model.kind == "3d"

    def test_a_preview_model_is_not_reported_as_saved(self):
        """It is in ComfyUI's temp directory, which ComfyUI empties."""
        found = outputs_from_history(HISTORY, GRAPH)
        preview = next(o for o in found if o.filename.startswith("preview3d_"))
        assert preview.type == "temp"
        assert not preview.is_saved

    def test_only_one_file_actually_survives_the_run(self):
        """Three outputs, one file. Counting all three is how a run claims to
        have made three models and leaves two of them nowhere."""
        saved = [o for o in outputs_from_history(HISTORY, GRAPH) if o.is_saved]
        assert [o.filename for o in saved] == ["ComfyUI_00001_.glb"]

    def test_the_mask_is_still_found_just_not_as_a_saved_file(self):
        found = outputs_from_history(HISTORY, GRAPH)
        mask = next(o for o in found if o.filename.endswith(".png"))
        assert mask.is_picture and not mask.is_saved

    def test_without_the_graph_nothing_gets_worse(self):
        """The graph is how previews are told from saves. Without it the old
        assumption stands -- wrong, but not a new failure."""
        found = outputs_from_history(HISTORY)
        assert all(o.is_saved for o in found if o.kind == "3d")

    def test_an_ordinary_picture_run_is_unaffected(self):
        entry = {"outputs": {"9": {"images": [
            {"filename": "ComfyUI_00007_.png", "subfolder": "", "type": "output"}]}}}
        found = outputs_from_history(entry, {"9": {"class_type": "SaveImage"}})
        assert len(found) == 1
        assert found[0].is_picture and found[0].is_saved


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6", reason="PySide6 is not installed")
    from PySide6 import QtWidgets

    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication(["tests"])


class TestWhatTheScreenSays:
    def page(self, tmp_path):
        from toolshed.ui.make import MakePage

        page = MakePage(tmp_path)
        page.client = None          # no engine; nothing here fetches a picture
        return page

    def test_the_model_is_named_and_its_real_folder_given(self, qapp, tmp_path):
        page = self.page(tmp_path)
        page._on_produced(outputs_from_history(HISTORY, GRAPH))
        said = page.status.text()
        assert "ComfyUI_00001_.glb" in said, said
        assert str(tmp_path / "output" / "3d") in said, said

    def test_previews_are_not_counted_as_things_you_made(self, qapp, tmp_path):
        page = self.page(tmp_path)
        page._on_produced(outputs_from_history(HISTORY, GRAPH))
        assert "1 file saved" in page.status.text(), page.status.text()

    def test_a_run_that_only_previewed_says_so(self, qapp, tmp_path):
        """Rather than pointing at an output folder that will stay empty."""
        page = self.page(tmp_path)
        page._on_produced([Output("preview3d_advanced_1.glb", "", "temp", "3d")])
        assert "nothing was saved" in page.status.text().lower()

    def test_a_picture_run_still_reads_as_it_did(self, qapp, tmp_path):
        page = self.page(tmp_path)
        page._on_produced([Output("ComfyUI_00007_.png", "", "output", "images")])
        said = page.status.text()
        assert "ComfyUI_00007_.png" in said and str(tmp_path / "output") in said

    def test_producing_nothing_is_unchanged(self, qapp, tmp_path):
        page = self.page(tmp_path)
        page._on_produced([])
        assert "produced no files" in page.status.text()


class TestOpeningTheFolder:
    def page(self, tmp_path):
        from toolshed.ui.make import MakePage

        page = MakePage(tmp_path)
        page.client = None
        return page

    def test_it_offers_the_folder_the_model_is_in(self, qapp, tmp_path):
        page = self.page(tmp_path)
        page._on_produced(outputs_from_history(HISTORY, GRAPH))
        assert page.output_dir_for(page._last_saved) == tmp_path / "output" / "3d"

    def test_with_nothing_made_yet_it_is_just_the_output_folder(self, qapp, tmp_path):
        page = self.page(tmp_path)
        assert page.output_dir_for(None) == tmp_path / "output"

    def test_a_temp_file_does_not_send_anyone_to_a_subfolder(self, qapp, tmp_path):
        """Its subfolder is meaningless outside ComfyUI's temp directory."""
        page = self.page(tmp_path)
        item = Output("preview3d_advanced_1.glb", "3d", "temp", "3d")
        assert page.output_dir_for(item) == tmp_path / "output"

    def test_a_dead_file_manager_leaves_the_path_on_screen(
            self, qapp, tmp_path, monkeypatch):
        """The reported bug: the button did nothing and said nothing. openUrl
        returning False was being discarded."""
        from PySide6 import QtGui

        page = self.page(tmp_path)
        monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", lambda _url: False)

        def no_opener(*_args, **_kwargs):
            raise OSError("no file manager")

        monkeypatch.setattr("subprocess.Popen", no_opener)
        page.open_folder()
        said = page.status.text()
        assert str(tmp_path / "output") in said
        assert "could not open" in said.lower()

    def test_the_platform_opener_is_tried_when_qt_declines(
            self, qapp, tmp_path, monkeypatch):
        """In a PyInstaller build Qt's own path is the one most likely missing,
        so a working xdg-open must still be used rather than given up on."""
        from PySide6 import QtGui

        page = self.page(tmp_path)
        monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", lambda _url: False)
        called: list[list[str]] = []
        monkeypatch.setattr("subprocess.Popen",
                            lambda cmd, **_kw: called.append(cmd) or None)
        page.open_folder()
        assert called and str(tmp_path / "output") in called[0][-1]
        assert "could not open" not in page.status.text().lower()


class TestMemoryFailuresThatDoNotSayOutOfMemory:
    """A maths library that cannot get a workspace reports its own status code.

    Verbatim from a 3D unwrap on a 20 GB gfx1100, after the model had already
    been built -- 3.04M vertices, 6.17M faces, 101 seconds in:

        CUDA error: HIPBLAS_STATUS_ALLOC_FAILED when calling
        `hipblasDgetrfBatched(handle, n, dA_array, ldda, ipiv_array, ...)`

    Nothing in that says "out of memory", so it used to reach the user exactly
    as written.
    """

    def explain(self, node_type: str, message: str) -> str:
        from toolshed.exec.comfy_api import _explain_execution_error

        return _explain_execution_error(
            {"node_type": node_type, "exception_message": message})

    HIPBLAS = ("CUDA error: HIPBLAS_STATUS_ALLOC_FAILED when calling "
               "`hipblasDgetrfBatched( handle, n, dA_array, ldda, ipiv_array, "
               "info_array, batchsize)`")

    def test_a_blas_allocation_failure_is_recognised(self):
        said = self.explain("UnwrapMesh", self.HIPBLAS)
        assert "ran out of memory" in said
        assert "HIPBLAS" not in said, "the raw status code is not an explanation"

    def test_the_mesh_steps_get_advice_that_applies_to_them(self):
        """"Ask for a smaller one" is meaningless when the photo decided the
        size and there is no size on screen."""
        said = self.explain("UnwrapMesh", self.HIPBLAS)
        assert "Decimate Mesh" in said or "Remesh Mesh" in said
        assert "picture or a video" not in said

    def test_the_cuda_spelling_is_caught_too(self):
        said = self.explain("KSampler", "CUBLAS_STATUS_ALLOC_FAILED")
        assert "ran out of memory" in said

    def test_a_picture_step_still_gets_picture_advice(self):
        said = self.explain("KSampler", "Allocation on device: out of memory")
        assert "picture or a video" in said
        assert "Decimate Mesh" not in said

    def test_an_unrelated_failure_is_not_dressed_up_as_memory(self):
        said = self.explain("LoadImage", "invalid file format")
        assert "ran out of memory" not in said
        assert "invalid file format" in said
