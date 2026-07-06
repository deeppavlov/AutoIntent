"""Pipeline.fit preflight integration: off / warn / strict modes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from autointent import Pipeline
from autointent._advisor import HardwareProfile, detect_hardware, run_preflight, stats_from_dataset_obj
from autointent._pipeline import PreflightError
from autointent.configs import LoggingConfig

if TYPE_CHECKING:
    from autointent import Dataset


def _tiny_hw() -> HardwareProfile:
    """Deterministic, intentionally-infeasible hardware budget."""
    return HardwareProfile(
        accelerator="cuda",
        device_name="test-tiny",
        vram_gb=0.1,
        ram_gb=0.5,
        free_disk_gb=1.0,
        cpu_count=2,
    )


def _classic_light_pipeline() -> Pipeline:
    p = Pipeline.from_preset("classic-light")
    p.set_config(LoggingConfig(dump_modules=False, clear_ram=True))
    return p


def test_preflight_off_skips_advisor(dataset: Dataset, caplog: pytest.LogCaptureFixture) -> None:
    """preflight='off' must not run the advisor (no Preflight log line)."""
    p = _classic_light_pipeline()
    with caplog.at_level(logging.INFO, logger="autointent._pipeline._pipeline"):
        try:
            p.fit(dataset, preflight="off")
        except Exception:  # noqa: BLE001 — fit may fail in test env; we only care about preflight side effect
            pass
    assert not any("Preflight" in r.getMessage() for r in caplog.records)


def test_preflight_warn_logs_findings(dataset: Dataset, caplog: pytest.LogCaptureFixture) -> None:
    """preflight='warn' logs a Preflight verdict line."""
    p = _classic_light_pipeline()
    with caplog.at_level(logging.INFO, logger="autointent._pipeline._pipeline"):
        try:
            p.fit(dataset, preflight="warn")
        except Exception:  # noqa: BLE001
            pass
    msgs = [r.getMessage() for r in caplog.records]
    assert any("Preflight" in m and "verdict=" in m for m in msgs)


def test_preflight_strict_raises_on_infeasible(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preflight='strict' raises PreflightError when findings include OVER.

    Forces a tiny hardware budget so even cheap presets blow it.
    """
    monkeypatch.setattr("autointent._pipeline._pipeline.detect_hardware", _tiny_hw)
    p = _classic_light_pipeline()
    with pytest.raises(PreflightError) as exc_info:
        p.fit(dataset, preflight="strict")
    assert exc_info.value.findings
    assert all(f.severity.value == "over" for f in exc_info.value.findings)


def test_preflight_warn_does_not_raise_on_infeasible(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Tiny hardware + warn mode logs an ERROR but doesn't raise."""
    monkeypatch.setattr("autointent._pipeline._pipeline.detect_hardware", _tiny_hw)
    p = _classic_light_pipeline()
    with caplog.at_level(logging.ERROR, logger="autointent._pipeline._pipeline"):
        try:
            p.fit(dataset, preflight="warn")
        except PreflightError:
            pytest.fail("warn mode must not raise PreflightError")
        except Exception:  # noqa: BLE001 — downstream fit errors are out of scope
            pass
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_pipeline_advisor_config_round_trip(dataset: Dataset) -> None:
    """End-to-end integration: Pipeline -> _build_advisor_config -> run_preflight.

    Asserts the round-trip is wired correctly: the dict ``Pipeline`` exposes to
    the advisor validates against ``OptimizationConfig``, the advisor produces a
    well-formed report, and the driver list reflects the actual modules from the
    preset's search space (not silently empty).
    """
    p = _classic_light_pipeline()
    config = p._build_advisor_config()  # noqa: SLF001
    stats = stats_from_dataset_obj(dataset)
    hardware = detect_hardware()

    report = run_preflight(config, stats, hardware, preset_name="classic-light")

    # The advisor accepted the pipeline-built config and produced findings.
    assert report.preset_name == "classic-light"
    assert report.resource.drivers, "expected at least one driver row for classic-light"

    # classic-light's scoring node has knn / linear / mlknn — at least linear
    # should always end up in drivers (knn variants don't always carry an
    # explicit model_name, so they're allowed to be absent).
    driver_modules = {d["module"] for d in report.resource.drivers}
    assert "linear" in driver_modules, f"missing linear scorer in drivers: {driver_modules}"

    # The advisor must always emit the three resource findings.
    metrics = {f.metric for f in report.findings if f.metric}
    assert {"vram", "ram", "disk"} <= metrics, f"missing required metrics: {metrics}"

    # Dataset stats round-trip into the report.
    assert report.dataset["n_samples"] == stats.n_samples
    assert report.dataset["n_classes"] == stats.n_classes
