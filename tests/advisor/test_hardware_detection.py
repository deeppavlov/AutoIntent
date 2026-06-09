"""Hardware detection has to be safe on every machine — broken CUDA, no GPU,
no psutil. Verify the fallbacks work without raising.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from autointent._advisor._hardware import detect_hardware


def test_cpu_fallback_when_no_accelerator() -> None:
    with (
        patch("autointent._advisor._hardware._detect_cuda", return_value=None),
        patch("autointent._advisor._hardware._detect_mps", return_value=None),
    ):
        hw = detect_hardware()
    assert hw.accelerator == "cpu"
    assert hw.vram_gb == 0.0
    assert hw.device_class == "cpu"


def test_cuda_branch_classifies_low_gpu() -> None:
    with (
        patch(
            "autointent._advisor._hardware._detect_cuda",
            return_value=(8.0, "NVIDIA RTX 3060"),
        ),
    ):
        hw = detect_hardware()
    assert hw.accelerator == "cuda"
    assert hw.vram_gb == pytest.approx(8.0)
    assert hw.device_class == "low-gpu"


def test_mps_budget_uses_ram_fraction() -> None:
    with (
        patch("autointent._advisor._hardware._detect_cuda", return_value=None),
        patch("autointent._advisor._hardware._detect_ram_gb", return_value=32.0),
        patch(
            "autointent._advisor._hardware._detect_mps",
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
            "autointent._advisor._hardware._detect_cuda",
            return_value=(24.0, "NVIDIA RTX 4090"),
        ),
    ):
        hw = detect_hardware(vram_budget_gb=8.0)
    assert hw.vram_gb == pytest.approx(8.0)
    assert any("manual VRAM budget" in n for n in hw.notes)


def test_broken_cuda_returns_none_does_not_crash() -> None:
    # _detect_cuda swallows torch quirks already; verify the wrapper holds.
    with (
        patch("autointent._advisor._hardware._detect_cuda", return_value=None),
        patch("autointent._advisor._hardware._detect_mps", return_value=None),
    ):
        hw = detect_hardware()
    assert hw.accelerator == "cpu"
