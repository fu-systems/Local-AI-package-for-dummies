"""Which PyTorch build to install, decided by a table rather than a heuristic.

Getting this wrong is the single most common failure in this space: the user
ends up with a CPU build, downloads 40 GB of models, and only then discovers
nothing can run. So the choice is explicit, tested against recorded hardware,
and verified immediately after installation rather than trusted.

Index URLs and driver floors are as researched on 2026-09-05 and are recorded
in docs/UPSTREAM.md. They are data, not logic, so they can be corrected without
touching this code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from toolshed.hw.detect import HardwareReport

# NVIDIA compute capability -> wheel index. cu130 covers Turing (7.5) through
# Blackwell (12.0); older cards need the cu126 line, which is the end of the
# road for them.
CUDA_MODERN = "https://download.pytorch.org/whl/cu130"
CUDA_LEGACY = "https://download.pytorch.org/whl/cu126"
ROCM = "https://download.pytorch.org/whl/rocm7.2"

# Driver floors, (linux, windows), for each CUDA line.
DRIVER_FLOOR = {
    CUDA_MODERN: ("580.65.06", "580.88"),
    CUDA_LEGACY: ("525.60.13", "528.33"),
}

# AMD targets ROCm supports officially. Anything else on RDNA2+ can usually be
# coaxed into working with an override, but that is the user's decision to
# make, not ours to make silently.
ROCM_OFFICIAL = frozenset({
    "gfx1200", "gfx1201",                      # RDNA4
    "gfx1100", "gfx1101", "gfx1102",           # RDNA3
    "gfx1030", "gfx1031", "gfx1032",           # RDNA2 (1031/1032 need an override)
    "gfx1150", "gfx1151", "gfx1152", "gfx1153",  # RDNA3.5 integrated
})
# Cards ROCm does not officially target but which usually work when told to
# pretend they are something else.
ROCM_OVERRIDE = {"gfx1031": "10.3.0", "gfx1032": "10.3.0"}


@dataclass(frozen=True)
class TorchChoice:
    """The decision, plus everything needed to explain or refuse it."""

    supported: bool
    index_url: str = ""
    env: dict[str, str] = field(default_factory=dict)
    # A stable identifier for the refusal, so tests and UI agree without
    # matching on prose.
    reason_key: str | None = None
    message: str = ""
    needs_consent: bool = False
    legacy: bool = False

    @property
    def expected_local_tag(self) -> str:
        """The marker that must appear in torch.__version__ afterwards.

        Verifying this is what turns "we think we installed a GPU build" into
        "we know we did".
        """
        if not self.index_url:
            return ""
        return "+" + self.index_url.rsplit("/", 1)[-1].replace(".", "")


def _version_at_least(have: str, want: str) -> bool:
    """Compare dotted driver versions numerically, tolerating junk."""
    def parts(v: str) -> list[int]:
        out = []
        for chunk in v.split("."):
            digits = "".join(c for c in chunk if c.isdigit())
            out.append(int(digits) if digits else 0)
        return out

    a, b = parts(have), parts(want)
    a += [0] * (len(b) - len(a))
    b += [0] * (len(a) - len(b))
    return a >= b


def choose_torch(report: HardwareReport) -> TorchChoice:
    """Pick the PyTorch build for this machine, or refuse with a reason."""
    gpu = report.primary
    if gpu is None or gpu.vendor == "unknown":
        return TorchChoice(False, reason_key="no_dgpu",
                           message="No dedicated graphics card was found.")

    if gpu.vendor == "intel":
        return TorchChoice(False, reason_key="intel_unsupported",
                           message="Intel graphics cards are not supported yet.")

    if gpu.vendor == "amd":
        if report.os != "linux":
            return TorchChoice(
                False, reason_key="amd_on_windows",
                message="AMD graphics cards are only supported on Linux.")
        if gpu.gfx and gpu.gfx not in ROCM_OFFICIAL:
            return TorchChoice(
                False, reason_key="amd_gpu_too_old",
                message=f"ROCm does not support this card ({gpu.gfx}).")
        env = {}
        needs_consent = False
        if gpu.gfx in ROCM_OVERRIDE:
            # Tell ROCm to treat the card as a supported sibling. It usually
            # works, but "usually" is the user's risk to accept, not ours.
            env["HSA_OVERRIDE_GFX_VERSION"] = ROCM_OVERRIDE[gpu.gfx]
            needs_consent = True
        return TorchChoice(True, index_url=ROCM, env=env, needs_consent=needs_consent,
                           message="AMD on Linux, using ROCm.")

    # NVIDIA
    cap = gpu.compute_capability
    if cap is not None and cap < 5.0:
        return TorchChoice(False, reason_key="nvidia_gpu_too_old",
                           message="This card is too old for current PyTorch builds.")
    legacy = cap is not None and cap < 7.5
    index = CUDA_LEGACY if legacy else CUDA_MODERN

    if gpu.driver_version:
        floor = DRIVER_FLOOR[index][0 if report.os == "linux" else 1]
        if not _version_at_least(gpu.driver_version, floor):
            return TorchChoice(
                False, reason_key="driver_too_old", index_url=index,
                message=f"Your graphics driver is {gpu.driver_version}. "
                        f"This card needs {floor} or newer.")

    return TorchChoice(
        True, index_url=index, legacy=legacy,
        message="NVIDIA, older CUDA build (this card is the end of the line)."
        if legacy else "NVIDIA, current CUDA build.")
