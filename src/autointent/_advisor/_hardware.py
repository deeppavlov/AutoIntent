"""Local hardware detection.

Probes CPU / RAM / disk and the highest-priority accelerator available
(CUDA → MPS → CPU). All probes are wrapped to fall back safely on a
broken install (e.g. CUDA driver mismatch) rather than crash the advisor.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
from dataclasses import dataclass, field
from typing import Literal

import psutil
import torch

logger = logging.getLogger(__name__)

Accelerator = Literal["cuda", "mps", "cpu"]

# matches macOS PYTORCH_MPS_HIGH_WATERMARK_RATIO default
MPS_DEFAULT_BUDGET_RATIO = 0.7


@dataclass
class HardwareProfile:
    accelerator: Accelerator
    device_name: str
    vram_gb: float
    ram_gb: float
    free_disk_gb: float
    cpu_count: int
    notes: list[str] = field(default_factory=list)

    @property
    def device_class(self) -> str:
        if self.accelerator == "cpu":
            return "cpu"
        if self.accelerator == "mps":
            return "apple-silicon"
        if self.vram_gb >= 24:
            return "high-gpu"
        if self.vram_gb >= 12:
            return "mid-gpu"
        return "low-gpu"


def _detect_ram_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)


def _detect_free_disk_gb(path: str | None = None) -> float:
    cache = path or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    probe_path = cache if os.path.exists(cache) else os.path.expanduser("~")
    try:
        usage = shutil.disk_usage(probe_path)
        return usage.free / (1024**3)
    except OSError as e:
        logger.debug("disk usage probe failed at %s: %s", probe_path, e)
        return 0.0


def _detect_cuda() -> tuple[float, str] | None:
    if not torch.cuda.is_available():
        return None
    idx = 0
    try:
        _free, total = torch.cuda.mem_get_info(idx)
        vram_gb = total / (1024**3)
    except (RuntimeError, AttributeError) as e:
        logger.debug("torch.cuda.mem_get_info failed: %s", e)
        return None
    name = torch.cuda.get_device_name(idx)
    return vram_gb, name


def _detect_mps(ram_gb: float, budget_ratio: float = MPS_DEFAULT_BUDGET_RATIO) -> tuple[float, str] | None:
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return None
    # apple silicon: unified memory; budget is fraction of total RAM
    return ram_gb * budget_ratio, f"Apple Silicon ({platform.machine()})"


def detect_hardware(
    *,
    vram_budget_gb: float | None = None,
    mps_budget_ratio: float = MPS_DEFAULT_BUDGET_RATIO,
) -> HardwareProfile:
    """Detect the local hardware, with optional manual overrides.

    Args:
        vram_budget_gb: when set, overrides the detected VRAM (use for
            shared-GPU machines where part of the device is taken).
        mps_budget_ratio: fraction of total RAM treated as the MPS
            "VRAM" budget on Apple Silicon.

    Returns:
        HardwareProfile reflecting current machine state.
    """
    notes: list[str] = []
    ram_gb = _detect_ram_gb()
    free_disk_gb = _detect_free_disk_gb()
    cpu_count = os.cpu_count() or 1

    cuda = _detect_cuda()
    if cuda is not None:
        vram_gb, device_name = cuda
        accel: Accelerator = "cuda"
    else:
        mps = _detect_mps(ram_gb, mps_budget_ratio)
        if mps is not None:
            vram_gb, device_name = mps
            accel = "mps"
            notes.append(f"MPS unified memory: VRAM budget = {mps_budget_ratio:.0%} of RAM.")
        else:
            vram_gb = 0.0
            device_name = platform.processor() or "cpu"
            accel = "cpu"

    if vram_budget_gb is not None:
        if vram_gb and vram_budget_gb > vram_gb:
            notes.append(f"Manual --budget-vram-gb={vram_budget_gb} exceeds detected {vram_gb:.1f} GB; using override.")
        notes.append(f"Using manual VRAM budget: {vram_budget_gb} GB.")
        vram_gb = vram_budget_gb

    return HardwareProfile(
        accelerator=accel,
        device_name=device_name,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        free_disk_gb=free_disk_gb,
        cpu_count=cpu_count,
        notes=notes,
    )
