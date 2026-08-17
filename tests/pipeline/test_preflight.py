"""Pipeline.fit preflight integration: default-off, warn, strict.

``fit()`` is driven with ``Pipeline._fit`` stubbed out, so these tests exercise
the preflight gate (which runs before any heavy work) without training anything.
With ``clear_ram=True, dump_modules=False`` the post-``_fit`` branch returns the
context immediately, so a stubbed ``_fit`` leaves ``fit()`` fully functional.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from autointent import Pipeline
from autointent.advisor import HardwareProfile, PreflightError, dataset_stats, detect_hardware, run_preflight
from autointent.configs import LoggingConfig

if TYPE_CHECKING:
    from autointent import Dataset

_PIPELINE_LOGGER = "autointent._pipeline._pipeline"


@pytest.fixture(autouse=True)
def _stub_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip optimization; every test here is about the gate that runs before it."""
    monkeypatch.setattr(Pipeline, "_fit", lambda _self, _context: None)


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


def test_fit_does_not_run_preflight_by_default(dataset: Dataset, caplog: pytest.LogCaptureFixture) -> None:
    """The default is opt-out: no preflight, no Hub round-trips, no log line."""
    p = _classic_light_pipeline()
    with caplog.at_level(logging.INFO, logger=_PIPELINE_LOGGER):
        p.fit(dataset)
    assert not any("Preflight" in r.getMessage() for r in caplog.records)


def test_preflight_off_skips_advisor(dataset: Dataset, caplog: pytest.LogCaptureFixture) -> None:
    p = _classic_light_pipeline()
    with caplog.at_level(logging.INFO, logger=_PIPELINE_LOGGER):
        p.fit(dataset, preflight="off")
    assert not any("Preflight" in r.getMessage() for r in caplog.records)


def test_preflight_warn_logs_verdict(dataset: Dataset, caplog: pytest.LogCaptureFixture) -> None:
    p = _classic_light_pipeline()
    with caplog.at_level(logging.INFO, logger=_PIPELINE_LOGGER):
        p.fit(dataset, preflight="warn")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("Preflight" in m and "verdict=" in m for m in msgs)


def test_preflight_strict_raises_on_infeasible(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched at the advisor, not the pipeline: the import is lazy now."""
    monkeypatch.setattr("autointent.advisor.detect_hardware", _tiny_hw)
    p = _classic_light_pipeline()
    with pytest.raises(PreflightError) as exc_info:
        p.fit(dataset, preflight="strict")
    assert exc_info.value.findings
    assert all(f.severity.value == "over" for f in exc_info.value.findings)


def test_preflight_warn_does_not_raise_on_infeasible(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("autointent.advisor.detect_hardware", _tiny_hw)
    p = _classic_light_pipeline()
    with caplog.at_level(logging.ERROR, logger=_PIPELINE_LOGGER):
        p.fit(dataset, preflight="warn")
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_importing_autointent_does_not_import_the_advisor() -> None:
    """The advisor pulls in huggingface_hub probes; it must stay off the import path.

    Checked in a subprocess because pytest has already imported the advisor into
    this process. Asserting on ``huggingface_hub`` itself would not work --
    ``datasets`` imports it regardless -- so the subpackage's own absence is the
    real invariant.
    """
    code = "import autointent, sys; print('autointent.advisor' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False", "importing autointent must not import autointent.advisor"


def test_pipeline_advisor_config_round_trip(dataset: Dataset) -> None:
    """End-to-end: Pipeline -> _build_advisor_config -> run_preflight."""
    p = _classic_light_pipeline()
    config = p._build_advisor_config()
    stats = dataset_stats(dataset)
    hardware = detect_hardware()

    report = run_preflight(config, stats, hardware, preset_name="classic-light")

    assert report.preset_name == "classic-light"
    assert report.resource.drivers, "expected at least one driver row for classic-light"

    driver_modules = {d["module"] for d in report.resource.drivers}
    assert "linear" in driver_modules, f"missing linear scorer in drivers: {driver_modules}"

    metrics = {f.metric for f in report.findings if f.metric}
    assert {"vram", "ram", "disk"} <= metrics, f"missing required metrics: {metrics}"

    assert report.dataset["n_samples"] == stats.n_samples
    assert report.dataset["n_classes"] == stats.n_classes
