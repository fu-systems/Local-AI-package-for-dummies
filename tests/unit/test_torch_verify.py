"""What we expect PyTorch to look like afterwards, and how wrong it has to be
before the install stops.

A healthy machine -- two AMD cards, ROCm 7.2, torch reporting
``{"available": true, "devices": 2}`` -- was refused at 26 percent with:

    The wrong PyTorch build was installed (2.14.0+rocm7.2, expected +rocm72).

Two separate mistakes, and the second is the worse one.

The tag was invented. expected_local_tag stripped dots from the index URL's
last segment, so ".../whl/rocm7.2" became "+rocm72", which is not a string
PyTorch has ever produced. CUDA segments contain no dots, so the bug could
only ever fire on AMD -- the one configuration there is hardware to test on.

And the severity was wrong. This step exists to catch a CPU-only build before
40 GB of models are downloaded onto it. A CPU build is caught by "no devices".
A GPU build with an unexpected tag is a working machine, and stopping there
protects nobody from anything.
"""

from __future__ import annotations

import json

import pytest

from toolshed.exec import uvtool
from toolshed.hw.detect import Gpu, HardwareReport
from toolshed.planner.torchsel import CUDA_LEGACY, CUDA_MODERN, ROCM, TorchChoice, choose_torch


class TestTheExpectedTag:
    """PyTorch names a wheel's local version after the index it is published
    under, so the index segment IS the tag -- dots and all."""

    @pytest.mark.parametrize("index, tag", [
        (ROCM, "+rocm7.2"),
        (CUDA_MODERN, "+cu130"),
        (CUDA_LEGACY, "+cu126"),
    ])
    def test_the_tag_is_the_index_segment_verbatim(self, index, tag):
        assert TorchChoice(True, index_url=index).expected_local_tag == tag

    def test_the_reported_rocm_version_satisfies_it(self):
        """The exact string from the install log that was rejected."""
        installed = "2.14.0+rocm7.2"
        expected = TorchChoice(True, index_url=ROCM).expected_local_tag
        assert expected in installed, f"{installed} should satisfy {expected}"

    def test_no_tag_ever_loses_a_character_of_its_segment(self):
        """The bug in one sentence: the tag was not what the URL said."""
        for index in (ROCM, CUDA_MODERN, CUDA_LEGACY):
            segment = index.rsplit("/", 1)[-1]
            assert TorchChoice(True, index_url=index).expected_local_tag == f"+{segment}"

    def test_an_amd_card_expects_a_tag_with_the_dot_in_it(self):
        report = HardwareReport(os="linux",
                                gpus=(Gpu("amd", "RX 7900 XT", 20480, gfx="gfx1100"),))
        assert choose_torch(report).expected_local_tag == "+rocm7.2"

    def test_no_index_means_no_expectation(self):
        assert TorchChoice(False).expected_local_tag == ""


def probe(monkeypatch, payload, *, ok=True, extra=""):
    """Stand in for running the probe inside the venv."""
    class Result:
        pass
    Result.ok = ok
    Result.stdout = extra + (json.dumps(payload) if payload is not None else "")
    Result.stderr = ""
    monkeypatch.setattr(uvtool, "run", lambda *a, **k: Result)


# The real payload from the reported machine. Note "devices": 2 with one
# graphics card: ROCm exposes the CPU as an HSA agent, and ComfyUI's own log
# shows it as
#     Device: cuda:0 AMD Radeon Graphics
#     Device: cuda:1 AMD Ryzen 7 7800X3D 8-Core Processor
WORKING_ROCM = {"version": "2.14.0+rocm7.2", "cuda": None, "hip": "7.2.53211",
                "available": True, "devices": 2, "name": "AMD Radeon Graphics"}


class TestWhatStopsTheInstall:
    def test_a_working_card_passes(self, monkeypatch, tmp_path):
        probe(monkeypatch, WORKING_ROCM)
        check = uvtool.verify_torch(tmp_path, "+rocm7.2")
        assert check.ok and not check.warning
        assert "AMD Radeon Graphics" in check.message

    def test_the_cpu_is_never_counted_as_a_graphics_card(self, monkeypatch, tmp_path):
        """torch.cuda.device_count() is 2 on this machine and one of them is a
        Ryzen. Reporting "2 graphics cards" was wrong, and it sent me on to
        suggest --cuda-device 1, which selects the processor."""
        probe(monkeypatch, WORKING_ROCM)
        message = uvtool.verify_torch(tmp_path, "+rocm7.2").message
        assert "2 graphics cards" not in message
        assert "cards" not in message, f"still pluralising a device count: {message}"

    def test_it_falls_back_gracefully_when_the_name_is_missing(self, monkeypatch, tmp_path):
        probe(monkeypatch, {**WORKING_ROCM, "name": ""})
        check = uvtool.verify_torch(tmp_path, "+rocm7.2")
        assert check.ok
        assert "your graphics card" in check.message

    def test_the_exact_reported_case_now_passes(self, monkeypatch, tmp_path):
        """Two cards, ROCm 7.2, torch happy. This must never stop an install
        again -- with the right tag, or with a wrong one."""
        probe(monkeypatch, WORKING_ROCM)
        report = HardwareReport(os="linux",
                                gpus=(Gpu("amd", "RX 7900 XT", 20480, gfx="gfx1100"),))
        assert uvtool.verify_torch(tmp_path, choose_torch(report).expected_local_tag).ok
        assert uvtool.verify_torch(tmp_path, "+rocm72").ok, \
            "even a wrong expectation must not reject a card that works"

    def test_a_cpu_only_build_stops_it(self, monkeypatch, tmp_path):
        """The failure this step exists for: torch installed, no card."""
        probe(monkeypatch, {"version": "2.14.0", "available": False, "devices": 0})
        check = uvtool.verify_torch(tmp_path, "+rocm7.2")
        assert not check.ok
        assert "cannot see your graphics card" in check.message

    def test_zero_devices_stops_it_even_if_available_is_true(self, monkeypatch, tmp_path):
        probe(monkeypatch, {"version": "2.14.0+rocm7.2", "available": True, "devices": 0})
        assert not uvtool.verify_torch(tmp_path, "+rocm7.2").ok

    def test_torch_that_will_not_import_stops_it(self, monkeypatch, tmp_path):
        probe(monkeypatch, None, ok=False)
        check = uvtool.verify_torch(tmp_path, "+rocm7.2")
        assert not check.ok
        assert "could not be loaded" in check.message

    def test_unreadable_output_stops_it(self, monkeypatch, tmp_path):
        probe(monkeypatch, None)
        assert not uvtool.verify_torch(tmp_path, "+rocm7.2").ok

    def test_an_unexpected_but_working_build_only_warns(self, monkeypatch, tmp_path):
        """A machine that works is not a failure. Say so and carry on."""
        probe(monkeypatch, WORKING_ROCM)
        check = uvtool.verify_torch(tmp_path, "+cu130")
        assert check.ok, "a working card was rejected over a version string"
        assert check.warning
        assert "carrying on" in check.warning
        assert "2.14.0+rocm7.2" in check.warning and "+cu130" in check.warning

    def test_noise_before_the_json_is_ignored(self, monkeypatch, tmp_path):
        """ROCm prints its own diagnostics to the console during import -- the
        reported log had four "(null): No such file or directory" lines before
        the payload. They are not ours and must not be read as failure."""
        probe(monkeypatch, WORKING_ROCM,
              extra="(null): No such file or directory\n" * 4)
        assert uvtool.verify_torch(tmp_path, "+rocm7.2").ok


class TestTheWarningIsRecorded:
    def test_a_mismatch_reaches_the_manifest_and_the_log(self, tmp_path, monkeypatch):
        """Logged so the user sees it now, recorded so it is still findable in
        a week when something behaves strangely."""
        from toolshed.exec.runner import Runner
        from toolshed.planner.plan import InstallPlan, Kind, Step

        probe(monkeypatch, WORKING_ROCM)
        report = HardwareReport(os="linux",
                                gpus=(Gpu("amd", "RX 7900 XT", 20480, gfx="gfx1100"),))
        step = Step(Kind.VERIFY_TORCH, "Checking your graphics card is really being used",
                    payload={"expect_tag": "+cu130"})
        plan = InstallPlan((step,), choose_torch(report), tmp_path)

        logs = []
        manifest = Runner(plan, on_event=lambda e: e.kind == "log" and logs.append(e.message)).run()

        assert any("carrying on" in n for n in manifest.notes)
        assert any("carrying on" in m for m in logs)

    def test_a_matching_build_records_nothing(self, tmp_path, monkeypatch):
        from toolshed.exec.runner import Runner
        from toolshed.planner.plan import InstallPlan, Kind, Step

        probe(monkeypatch, WORKING_ROCM)
        report = HardwareReport(os="linux",
                                gpus=(Gpu("amd", "RX 7900 XT", 20480, gfx="gfx1100"),))
        step = Step(Kind.VERIFY_TORCH, "Checking", payload={"expect_tag": "+rocm7.2"})
        manifest = Runner(InstallPlan((step,), choose_torch(report), tmp_path)).run()
        assert manifest.notes == []
