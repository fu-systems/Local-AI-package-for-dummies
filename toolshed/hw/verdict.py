"""Turn a HardwareReport into what we actually tell the user.

This module is a **pure function**: no I/O, no subprocesses, no filesystem. That
is deliberate and load-bearing. It means every hardware verdict -- including for
cards nobody on the project owns -- is provable on a machine with no GPU, from a
recorded HardwareReport fixture. Given that the only test hardware is one AMD
Linux box, this is how the NVIDIA and Windows halves of the product get tested
at all.

The supported set is fixed by decision D2 in docs/PLAN.md: NVIDIA on Windows and
Linux, AMD on Linux only. Everything else gets a clear refusal rather than an
install that will not work.
"""

from __future__ import annotations

from dataclasses import dataclass

from toolshed.hw.detect import Gpu, HardwareReport

# VRAM floors in GB. Sourced from the official ComfyUI templates' own totals;
# see catalog/recipes/*.generated.yaml and docs/PLAN.md section 9.
VRAM_PICTURES = 6.0
VRAM_VIDEO = 8.0
VRAM_MUSIC = 8.0
VRAM_3D = 10.0

MODALITIES = ("Pictures", "Video", "Music", "3D models")


@dataclass(frozen=True)
class Verdict:
    supported: bool
    headline: str
    detail: tuple[str, ...] = ()
    # Stable identifier for the refusal, so the UI and the tests agree on which
    # case fired without matching on prose that will be rewritten.
    reason_key: str | None = None
    modalities: tuple[tuple[str, bool, str], ...] = ()

    @property
    def offered(self) -> tuple[str, ...]:
        return tuple(name for name, ok, _ in self.modalities if ok)


def _modalities_for(vram_gb: float | None, *, amd: bool) -> tuple[tuple[str, bool, str], ...]:
    if vram_gb is None:
        unknown = "We could not read your card's memory, so we will check during setup."
        return tuple((name, True, unknown) for name in MODALITIES)

    three_d_note = (
        "3D on AMD is experimental. We will test it during setup and tell you honestly."
        if amd
        else "Turns a photo into a 3D model."
    )
    return (
        ("Pictures", vram_gb >= VRAM_PICTURES,
         "A few seconds each." if vram_gb >= VRAM_PICTURES
         else f"Needs at least {VRAM_PICTURES:g} GB of graphics memory."),
        ("Video", vram_gb >= VRAM_VIDEO,
         "Short clips, a few minutes each." if vram_gb >= VRAM_VIDEO
         else f"Needs at least {VRAM_VIDEO:g} GB of graphics memory."),
        ("Music", vram_gb >= VRAM_MUSIC,
         "Full songs from a description." if vram_gb >= VRAM_MUSIC
         else f"Needs at least {VRAM_MUSIC:g} GB of graphics memory."),
        ("3D models", vram_gb >= VRAM_3D,
         three_d_note if vram_gb >= VRAM_3D
         else f"Needs at least {VRAM_3D:g} GB of graphics memory."),
    )


def _supported(gpu: Gpu, report: HardwareReport) -> Verdict:
    vram = gpu.vram_gb
    amd = gpu.vendor == "amd"
    modalities = _modalities_for(vram, amd=amd)

    if vram is not None and vram < VRAM_PICTURES:
        return Verdict(
            supported=False,
            headline=f"Your {gpu.describe()} does not have enough memory.",
            detail=(
                f"Making pictures needs about {VRAM_PICTURES:g} GB of graphics memory "
                f"and this card has {vram:g} GB.",
                "We would rather tell you now than have you download 12 GB and find out.",
            ),
            reason_key="vram_below_min",
            modalities=modalities,
        )

    if amd:
        headline = f"Your {gpu.describe()} on Linux will work."
    else:
        headline = f"Good news — your {gpu.describe()} will work well."

    detail: list[str] = []
    if vram is None:
        detail.append("We could not read how much memory this card has, so we will "
                      "check properly during setup.")
    if amd:
        detail.append("AMD cards are supported on Linux through ROCm.")
    return Verdict(supported=True, headline=headline, detail=tuple(detail),
                   modalities=modalities)


def verdict_for(report: HardwareReport) -> Verdict:
    """Decide what this machine can do, and how to say it."""
    if report.os not in {"linux", "windows"}:
        return Verdict(
            supported=False,
            headline="Toolshed runs on Windows and Linux.",
            detail=("This computer is running something else, so we cannot set it up here.",),
            reason_key="os_unsupported",
        )

    gpu = report.primary
    if gpu is None or gpu.vendor == "unknown":
        return Verdict(
            supported=False,
            headline="You need a dedicated graphics card.",
            detail=(
                "We could not find one in this computer.",
                "On the processor alone a single picture takes 10 to 20 minutes and "
                "video is out of the question, so we do not offer it.",
            ),
            reason_key="no_dgpu",
        )

    if gpu.vendor == "intel":
        return Verdict(
            supported=False,
            headline="We do not support Intel graphics cards yet.",
            detail=(
                "It is technically possible and it is on our list.",
                "We would rather tell you now than have you download 40 GB and find out.",
            ),
            reason_key="intel_unsupported",
        )

    if gpu.vendor == "amd" and report.os == "windows":
        return Verdict(
            supported=False,
            headline="We cannot set this up on Windows with an AMD graphics card yet.",
            detail=(
                "AMD's own Windows support for this is still an early preview, and the "
                "standard downloads do not include Windows AMD builds, so we cannot set "
                "it up reliably.",
                "It works well on Linux, and we support that.",
                "If this machine also has an NVIDIA card, we can use that instead.",
            ),
            reason_key="amd_on_windows",
        )

    if gpu.vendor in {"nvidia", "amd"}:
        return _supported(gpu, report)

    return Verdict(
        supported=False,
        headline="We do not recognise this graphics card.",
        detail=("Tell us about your setup and we will look into it.",),
        reason_key="vendor_unknown",
    )
