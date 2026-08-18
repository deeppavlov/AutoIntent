"""Unit tests for ``_pick_module_to_drop``'s constraint selection.

Built from hand-rolled reports rather than real preflight runs: the rule under
test is "prune the module heaviest along the dimension that is actually over
budget", and that rule should be verifiable without invoking any formula.

Regression guard for experiments #40 finding 3 — the metric-name mismatch that
made every prune a VRAM prune.
"""

from __future__ import annotations

from autointent.advisor._report import PreflightReport, Severity
from autointent.advisor._workflows import _pick_module_to_drop


def _report(*over_metrics: str) -> PreflightReport:
    """Report whose scoring drivers disagree about which module is heaviest.

    ``bert`` is heaviest on VRAM, ``linear`` on RAM, ``catboost`` on time — so
    the module returned identifies which dimension the code actually consulted.
    """
    report = PreflightReport()
    for metric in ("vram", "ram", "disk", "time"):
        severity = Severity.OVER if metric in over_metrics else Severity.AMPLE
        report.add("resource", severity, f"{metric} finding", metric=metric)
    report.resource.drivers = [
        {"node_type": "scoring", "module": "bert", "vram_gb": 20.0, "ram_gb": 3.0, "time_hours": 2.0},
        {"node_type": "scoring", "module": "linear", "vram_gb": 0.5, "ram_gb": 40.0, "time_hours": 1.0},
        {"node_type": "scoring", "module": "catboost", "vram_gb": 0.2, "ram_gb": 8.0, "time_hours": 90.0},
        # Decision modules are never droppable, however heavy they look.
        {"node_type": "decision", "module": "argmax", "vram_gb": 99.0, "ram_gb": 99.0, "time_hours": 99.0},
    ]
    return report


def test_ram_over_prunes_ram_heaviest() -> None:
    assert _pick_module_to_drop(_report("ram")) == ("scoring", "linear")


def test_time_over_prunes_time_heaviest() -> None:
    assert _pick_module_to_drop(_report("time")) == ("scoring", "catboost")


def test_vram_over_prunes_vram_heaviest() -> None:
    assert _pick_module_to_drop(_report("vram")) == ("scoring", "bert")


def test_vram_wins_when_several_constraints_are_over() -> None:
    """Documented preference order is VRAM > time > RAM."""
    assert _pick_module_to_drop(_report("vram", "ram", "time")) == ("scoring", "bert")


def test_time_beats_ram_when_both_over() -> None:
    assert _pick_module_to_drop(_report("ram", "time")) == ("scoring", "catboost")


def test_disk_over_falls_back_to_vram_proxy() -> None:
    """Drivers carry no per-module disk figure, so disk reduces by the VRAM proxy."""
    assert _pick_module_to_drop(_report("disk")) == ("scoring", "bert")


def test_no_over_findings_falls_back_to_vram() -> None:
    assert _pick_module_to_drop(_report()) == ("scoring", "bert")


def test_returns_none_when_no_scoring_driver_is_droppable() -> None:
    report = _report("ram")
    report.resource.drivers = [
        {"node_type": "decision", "module": "argmax", "vram_gb": 1.0, "ram_gb": 1.0, "time_hours": 1.0},
        {"node_type": "scoring", "module": "?", "vram_gb": 1.0, "ram_gb": 1.0, "time_hours": 1.0},
    ]
    assert _pick_module_to_drop(report) is None
