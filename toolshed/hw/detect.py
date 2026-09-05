"""Detect the machine's GPUs.

Two hard rules, both learned from other people's bug trackers:

* **Never import torch here.** This runs before any virtual environment exists.
* **Never trust ``Win32_VideoController.AdapterRAM``.** It is a uint32 and
  saturates at 4294967295 on every modern card, and ``wmic.exe`` -- which most
  snippets reach for -- was removed outright by KB5067470 on current Windows.

Detection is best-effort by design: a probe that is missing or fails yields less
information rather than an exception. The caller renders whatever came back, and
:mod:`toolshed.hw.verdict` decides what it means.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

# PCI vendor IDs, as they appear in /sys/class/drm/*/device/vendor.
PCI_VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel"}

_PROBE_TIMEOUT = 10


@dataclass(frozen=True)
class Gpu:
    vendor: str = "unknown"          # nvidia | amd | intel | unknown
    name: str = ""
    vram_mb: int | None = None
    driver_version: str = ""
    gfx: str = ""                    # AMD LLVM target, e.g. gfx1100
    discrete: bool = True

    @property
    def vram_gb(self) -> float | None:
        return None if self.vram_mb is None else round(self.vram_mb / 1024, 1)

    def describe(self) -> str:
        parts = [self.name or f"{self.vendor} GPU"]
        if self.vram_gb is not None:
            parts.append(f"({self.vram_gb:g} GB)")
        return " ".join(parts)


@dataclass(frozen=True)
class HardwareReport:
    os: str = "unknown"              # linux | windows | other
    gpus: tuple[Gpu, ...] = ()
    notes: tuple[str, ...] = field(default=())

    @property
    def primary(self) -> Gpu | None:
        """The GPU we would actually use.

        Discrete cards win over integrated ones, and the one with the most VRAM
        wins among those. VRAM is never summed across cards -- two 8 GB cards
        are not a 16 GB card, and treating them as one is how a user ends up
        being offered a model that cannot load.
        """
        candidates = [g for g in self.gpus if g.discrete] or list(self.gpus)
        if not candidates:
            return None
        return max(candidates, key=lambda g: (g.vram_mb or 0))


def _run(cmd: list[str]) -> str | None:
    """Run a probe, returning its stdout, or None if it is unavailable."""
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


# --------------------------------------------------------------------------
# NVIDIA -- same probe on both operating systems
# --------------------------------------------------------------------------

def _nvidia_smi() -> list[Gpu]:
    query = "name,memory.total,driver_version"
    out = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if out is None:
        # Older drivers reject unknown fields wholesale rather than ignoring
        # them, so a narrower query is worth one retry before giving up.
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        if out is None:
            return []

    gpus: list[Gpu] = []
    for line in out.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 2:
            continue
        try:
            vram = int(float(cols[1]))
        except ValueError:
            vram = None
        gpus.append(
            Gpu(
                vendor="nvidia",
                name=cols[0],
                vram_mb=vram,
                driver_version=cols[2] if len(cols) > 2 else "",
            )
        )
    return gpus


# --------------------------------------------------------------------------
# Linux
# --------------------------------------------------------------------------

def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _amd_gfx(device_dir: Path) -> str:
    """Resolve an AMD card's LLVM target (gfx1100 and friends)."""
    # The kernel exposes it directly on newer amdgpu builds.
    for candidate in ("gfx_target_version", "device/gfx_target_version"):
        raw = _read(device_dir / candidate)
        if raw.isdigit() and len(raw) >= 5:
            # The kernel encodes gfx1100 as 110000: major/minor/step, two
            # digits each, with the trailing step digits dropped for the name.
            return f"gfx{raw[:-2]}"
    out = _run(["rocminfo"])
    if out:
        m = re.search(r"\b(gfx\d+[a-f0-9]*)\b", out)
        if m:
            return m.group(1)
    return ""


def _linux_gpus() -> tuple[list[Gpu], list[str]]:
    gpus: list[Gpu] = []
    notes: list[str] = []
    drm = Path("/sys/class/drm")
    if not drm.is_dir():
        return gpus, ["/sys/class/drm is not present; cannot enumerate GPUs"]

    for card in sorted(drm.glob("card[0-9]*")):
        device = card / "device"
        raw_vendor = _read(device / "vendor")
        try:
            vendor_id = int(raw_vendor, 16)
        except ValueError:
            continue
        vendor = PCI_VENDORS.get(vendor_id, "unknown")

        vram_mb: int | None = None
        # amdgpu reports exact bytes. This is the only fully reliable VRAM
        # number available anywhere in this module.
        raw_vram = _read(device / "mem_info_vram_total")
        if raw_vram.isdigit():
            vram_mb = int(raw_vram) // (1024 * 1024)

        gfx = _amd_gfx(device) if vendor == "amd" else ""
        # An integrated GPU shares system memory; amdgpu iGPUs report a small
        # carve-out that must not be read as a real card.
        discrete = not (vendor == "amd" and gfx.startswith("gfx11") and (vram_mb or 0) < 1024)

        gpus.append(Gpu(vendor=vendor, name=_read(device / "label") or "", vram_mb=vram_mb,
                        gfx=gfx, discrete=discrete))

    # NVIDIA does not publish VRAM through sysfs, so merge in nvidia-smi.
    nvidia = _nvidia_smi()
    if nvidia:
        gpus = [g for g in gpus if g.vendor != "nvidia"] + nvidia
        driver = _read(Path("/sys/module/nvidia/version"))
        if driver:
            gpus = [replace(g, driver_version=g.driver_version or driver)
                    if g.vendor == "nvidia" else g for g in gpus]
    elif any(g.vendor == "nvidia" for g in gpus):
        notes.append("An NVIDIA card is present but nvidia-smi did not run; "
                     "the driver may not be installed.")
    return gpus, notes


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------

def _windows_registry_gpus() -> list[Gpu]:
    """Read adapter names and VRAM from the display class registry key.

    ``HardwareInformation.qwMemorySize`` is a REG_QWORD and is the value that
    does *not* saturate at 4 GB, unlike the AdapterRAM property every WMI
    snippet on the internet reaches for.
    """
    try:
        import winreg  # noqa: PLC0415 -- Windows only, imported lazily on purpose
    except ImportError:
        return []

    key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus: list[Gpu] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as root:
            index = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as adapter:
                        def value(name: str):
                            try:
                                return winreg.QueryValueEx(adapter, name)[0]
                            except OSError:
                                return None

                        name = value("DriverDesc") or ""
                        if not name:
                            continue
                        qw = value("HardwareInformation.qwMemorySize")
                        vram_mb = int(qw) // (1024 * 1024) if qw else None
                        lowered = name.lower()
                        vendor = ("nvidia" if "nvidia" in lowered or "geforce" in lowered
                                  else "amd" if "amd" in lowered or "radeon" in lowered
                                  else "intel" if "intel" in lowered else "unknown")
                        gpus.append(Gpu(vendor=vendor, name=name, vram_mb=vram_mb,
                                        driver_version=str(value("DriverVersion") or ""),
                                        discrete=vendor != "intel"))
                except OSError:
                    continue
    except OSError:
        return []
    return gpus


def _windows_gpus() -> tuple[list[Gpu], list[str]]:
    gpus = _nvidia_smi()
    registry = _windows_registry_gpus()
    # Keep nvidia-smi's numbers where we have them; it is authoritative for
    # NVIDIA. Fill in every other adapter from the registry.
    seen_nvidia = bool(gpus)
    for gpu in registry:
        if gpu.vendor == "nvidia" and seen_nvidia:
            continue
        gpus.append(gpu)
    notes: list[str] = []
    if not gpus:
        notes.append("No display adapters were found in the registry.")
    return gpus, notes


# --------------------------------------------------------------------------

def detect() -> HardwareReport:
    """Probe this machine. Never raises."""
    system = platform.system().lower()
    if system == "linux":
        gpus, notes = _linux_gpus()
        os_name = "linux"
    elif system == "windows":
        gpus, notes = _windows_gpus()
        os_name = "windows"
    else:
        gpus, notes = [], [f"{platform.system()} is not a supported operating system."]
        os_name = "other"

    if not gpus and os_name in {"linux", "windows"}:
        notes.append("No dedicated graphics card was detected.")

    return HardwareReport(os=os_name, gpus=tuple(gpus), notes=tuple(notes))


if __name__ == "__main__":  # pragma: no cover -- manual probe
    report = detect()
    print(f"os={report.os}")
    for gpu in report.gpus:
        print(f"  {gpu.vendor:8s} {gpu.describe()} gfx={gpu.gfx or '-'} "
              f"driver={gpu.driver_version or '-'} discrete={gpu.discrete}")
    for note in report.notes:
        print(f"  note: {note}")
    sys.exit(0)
