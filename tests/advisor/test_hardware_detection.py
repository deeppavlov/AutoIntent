"""Accelerator selection in ``detect_hardware``: CUDA -> MPS -> CPU.

Each test patches ``_detect_cuda`` / ``_detect_mps`` (and sometimes
``_detect_ram_gb``) to force one branch, then checks the resulting profile —
the CPU fallback when nothing is available, the device_class thresholds, the
MPS unified-memory budget, and the manual VRAM override.

These do *not* cover a missing ``psutil``: it is a core dependency, imported
unguarded at ``_hardware.py`` module level, and no psutil-absent fallback
exists. The RAM and disk probes are therefore always the real ones.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from autointent.advisor._hardware import detect_hardware


def test_cpu_fallback_when_no_accelerator() -> None:
    with (
        patch("autointent.advisor._hardware._detect_cuda", return_value=None),
        patch("autointent.advisor._hardware._detect_mps", return_value=None),
    ):
        hw = detect_hardware()
    assert hw.accelerator == "cpu"
    assert hw.vram_gb == 0.0
    assert hw.device_class == "cpu"


def test_cuda_branch_classifies_low_gpu() -> None:
    with (
        patch(
            "autointent.advisor._hardware._detect_cuda",
            return_value=(8.0, "NVIDIA RTX 3060"),
        ),
    ):
        hw = detect_hardware()
    assert hw.accelerator == "cuda"
    assert hw.vram_gb == pytest.approx(8.0)
    assert hw.device_class == "low-gpu"


def test_mps_budget_uses_ram_fraction() -> None:
    with (
        patch("autointent.advisor._hardware._detect_cuda", return_value=None),
        patch("autointent.advisor._hardware._detect_ram_gb", return_value=32.0),
        patch(
            "autointent.advisor._hardware._detect_mps",
            side_effect=lambda ram, ratio: (ram * ratio, "Apple Silicon (arm64)"),
        ),
    ):
        hw = detect_hardware()
    assert hw.accelerator == "mps"
    assert hw.vram_gb == pytest.approx(32.0 * 0.7)
    assert any("MPS unified memory" in n for n in hw.notes)


def test_vram_budget_override_applies() -> None:
    with (
        patch(
            "autointent.advisor._hardware._detect_cuda",
            return_value=(24.0, "NVIDIA RTX 4090"),
        ),
    ):
        hw = detect_hardware(vram_budget_gb=8.0)
    assert hw.vram_gb == pytest.approx(8.0)
    assert any("manual VRAM budget" in n for n in hw.notes)


def test_broken_cuda_returns_none_does_not_crash() -> None:
    # _detect_cuda swallows torch quirks already; verify the wrapper holds.
    with (
        patch("autointent.advisor._hardware._detect_cuda", return_value=None),
        patch("autointent.advisor._hardware._detect_mps", return_value=None),
    ):
        hw = detect_hardware()
    assert hw.accelerator == "cpu"
