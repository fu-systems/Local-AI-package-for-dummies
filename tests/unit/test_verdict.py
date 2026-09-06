"""Tests for the hardware verdict.

`verdict_for` is a pure function of a HardwareReport, which is the entire point:
the project's only test hardware is one AMD Linux box, so every NVIDIA verdict,
every Windows verdict and all four D2 refusals are proven here from recorded
reports rather than from a card nobody owns.

Assertions are on `reason_key`, never on prose. The wording of every message
here will be rewritten many times; the decision it encodes should not silently
change with it.
"""

from __future__ import annotations

import pytest

from toolshed.hw.detect import Gpu, HardwareReport
from toolshed.hw.verdict import verdict_for


def report(os_name: str, *gpus: Gpu) -> HardwareReport:
    return HardwareReport(os=os_name, gpus=tuple(gpus))


NVIDIA_4070 = Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 4070", vram_mb=12288,
                  driver_version="580.88")
NVIDIA_3060_8G = Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 3060", vram_mb=8192)
NVIDIA_1660_6G = Gpu(vendor="nvidia", name="NVIDIA GeForce GTX 1660", vram_mb=6144)
NVIDIA_1050_4G = Gpu(vendor="nvidia", name="NVIDIA GeForce GTX 1050", vram_mb=4096)
NVIDIA_4090 = Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 4090", vram_mb=24576)
AMD_7900XTX = Gpu(vendor="amd", name="Radeon RX 7900 XTX", vram_mb=24576, gfx="gfx1100")
AMD_6700XT = Gpu(vendor="amd", name="Radeon RX 6700 XT", vram_mb=12288, gfx="gfx1031")
INTEL_ARC = Gpu(vendor="intel", name="Intel Arc A770", vram_mb=16384)
INTEL_IGPU = Gpu(vendor="intel", name="Intel UHD Graphics", vram_mb=128, discrete=False)


class TestD2Refusals:
    """The four cases decision D2 says we refuse rather than half-support."""

    def test_amd_on_windows_is_refused(self):
        v = verdict_for(report("windows", AMD_7900XTX))
        assert not v.supported
        assert v.reason_key == "amd_on_windows"

    def test_amd_on_linux_is_supported(self):
        """Same card, different OS. This pair is the whole of D2 in two tests."""
        v = verdict_for(report("linux", AMD_7900XTX))
        assert v.supported
        assert v.reason_key is None

    @pytest.mark.parametrize("os_name", ["linux", "windows"])
    def test_intel_is_refused(self, os_name):
        v = verdict_for(report(os_name, INTEL_ARC))
        assert not v.supported
        assert v.reason_key == "intel_unsupported"

    @pytest.mark.parametrize("os_name", ["linux", "windows"])
    def test_no_discrete_gpu_is_refused_not_downgraded(self, os_name):
        """We refuse rather than silently installing a CPU build. A single
        picture on the processor alone takes 10-20 minutes."""
        v = verdict_for(report(os_name))
        assert not v.supported
        assert v.reason_key == "no_dgpu"

    def test_macos_is_refused(self):
        v = verdict_for(report("other", NVIDIA_4070))
        assert not v.supported
        assert v.reason_key == "os_unsupported"


class TestVramTiers:
    def test_card_below_the_picture_floor_is_refused_early(self):
        v = verdict_for(report("windows", NVIDIA_1050_4G))
        assert not v.supported
        assert v.reason_key == "vram_below_min"

    def test_six_gb_gets_pictures_only(self):
        v = verdict_for(report("windows", NVIDIA_1660_6G))
        assert v.supported
        assert v.offered == ("Pictures",)

    def test_eight_gb_adds_video_and_music_but_not_3d(self):
        v = verdict_for(report("linux", NVIDIA_3060_8G))
        assert set(v.offered) == {"Pictures", "Video", "Music"}
        assert "3D models" not in v.offered

    def test_twelve_gb_gets_everything(self):
        v = verdict_for(report("windows", NVIDIA_4070))
        assert set(v.offered) == {"Pictures", "Video", "Music", "3D models"}

    def test_unknown_vram_defers_rather_than_refusing(self):
        """Failing to read VRAM must not lock someone out of their own card.
        Setup's smoke test settles it later."""
        v = verdict_for(report("linux", Gpu(vendor="nvidia", name="RTX ????", vram_mb=None)))
        assert v.supported
        assert len(v.offered) == 4


class TestPrimaryGpuSelection:
    def test_discrete_wins_over_integrated(self):
        v = verdict_for(report("linux", INTEL_IGPU, NVIDIA_4070))
        assert v.supported, "an Intel iGPU alongside a real card must not trigger the Intel refusal"

    def test_vram_is_never_summed_across_cards(self):
        """Two 8 GB cards are not a 16 GB card. Summing them offers models that
        cannot load."""
        r = report("linux", NVIDIA_3060_8G, NVIDIA_3060_8G)
        assert r.primary.vram_gb == 8.0
        assert "3D models" not in verdict_for(r).offered

    def test_largest_discrete_card_is_chosen(self):
        r = report("linux", NVIDIA_3060_8G, NVIDIA_4090)
        assert r.primary.vram_mb == 24576

    def test_integrated_only_machine_is_refused(self):
        v = verdict_for(report("linux", INTEL_IGPU))
        assert not v.supported


class TestMessaging:
    def test_amd_3d_is_labelled_experimental(self):
        """Nobody has published a working TRELLIS.2-on-ROCm result, so the UI
        must not imply we know it works."""
        v = verdict_for(report("linux", AMD_7900XTX))
        note = next(n for name, _, n in v.modalities if name == "3D models")
        assert "experimental" in note.lower()

    def test_nvidia_3d_is_not_labelled_experimental(self):
        v = verdict_for(report("windows", NVIDIA_4070))
        note = next(n for name, _, n in v.modalities if name == "3D models")
        assert "experimental" not in note.lower()

    def test_every_refusal_explains_itself(self):
        for r in [report("windows", AMD_7900XTX), report("linux", INTEL_ARC),
                  report("linux"), report("windows", NVIDIA_1050_4G)]:
            v = verdict_for(r)
            assert not v.supported
            assert v.headline and v.detail, f"{v.reason_key} gives the user nothing to act on"

    def test_unofficial_amd_card_is_still_offered(self):
        """gfx1031 is not officially supported by ROCm but usually works with an
        override. That is a consent decision at install time, not a refusal here."""
        assert verdict_for(report("linux", AMD_6700XT)).supported


class TestDesktopFileDetection:
    """The XDG portal warning a user sees when running the portable tarball.

    Qt registers the desktop file name with the portal; when no matching
    .desktop is installed the portal answers with a DBus error on stderr. We
    only claim the name when the file is really there.
    """

    def _fn(self):
        # toolshed.desktop, not toolshed.ui.app: this is a filesystem question
        # and must stay importable without Qt, or CI (which has no PySide6)
        # cannot run it.
        from toolshed.desktop import desktop_file_installed

        return desktop_file_installed

    def test_absent_when_no_desktop_file_anywhere(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("XDG_DATA_DIRS", str(tmp_path / "a") + ":" + str(tmp_path / "b"))
        assert self._fn()("toolshed") is False

    def test_found_in_xdg_data_home(self, tmp_path, monkeypatch):
        apps = tmp_path / "home" / "applications"
        apps.mkdir(parents=True)
        (apps / "toolshed.desktop").write_text("[Desktop Entry]\n")
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("XDG_DATA_DIRS", str(tmp_path / "nowhere"))
        assert self._fn()("toolshed") is True

    def test_found_in_a_later_xdg_data_dir(self, tmp_path, monkeypatch):
        apps = tmp_path / "b" / "applications"
        apps.mkdir(parents=True)
        (apps / "toolshed.desktop").write_text("[Desktop Entry]\n")
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("XDG_DATA_DIRS", f"{tmp_path / 'a'}:{tmp_path / 'b'}")
        assert self._fn()("toolshed") is True

    def test_non_linux_is_unaffected(self, monkeypatch):
        """Only the XDG portal cares. Windows must not lose the desktop name."""
        monkeypatch.setattr("sys.platform", "win32")
        assert self._fn()("toolshed") is True
