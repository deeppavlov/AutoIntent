"""Unit tests for the report dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from autointent._advisor._report import (
    DatasetStats,
    Finding,
    PreflightReport,
    ResourceEstimate,
    Severity,
)


class TestSeverityOrdering:
    def test_headroom_on_empty_report_is_green(self) -> None:
        assert PreflightReport().headroom == Severity.AMPLE

    def test_red_beats_yellow_beats_green(self) -> None:
        r = PreflightReport()
        r.add("resource", Severity.AMPLE, "ok")
        r.add("data", Severity.TIGHT, "warn")
        assert r.headroom == Severity.TIGHT
        r.add("config", Severity.OVER, "fail")
        assert r.headroom == Severity.OVER  # type: ignore[comparison-overlap]

    def test_is_feasible_flips_on_any_red(self) -> None:
        r = PreflightReport()
        r.add("resource", Severity.TIGHT, "warn")
        assert r.is_feasible is True
        r.add("data", Severity.OVER, "fail")
        assert r.is_feasible is False


class TestDatasetStatsPlaceholder:
    def test_defaults_populate_p95_above_avg(self) -> None:
        stats = DatasetStats.placeholder()
        assert stats.n_samples == 1_000
        assert stats.p95_tokens is not None
        assert stats.p95_tokens > stats.avg_tokens
        assert stats.source == "placeholder"

    def test_overrides_propagate(self) -> None:
        stats = DatasetStats.placeholder(n_samples=42, n_classes=3, avg_tokens=80, multilabel=True)
        assert stats.n_samples == 42
        assert stats.n_classes == 3
        assert stats.avg_tokens == 80
        assert stats.multilabel is True


class TestResourceEstimate:
    def test_total_disk_sums_download_and_dump(self) -> None:
        e = ResourceEstimate(disk_download_gb=2.5, disk_dump_gb=4.0)
        assert e.total_disk_gb == pytest.approx(6.5)

    def test_total_disk_ignores_cached(self) -> None:
        e = ResourceEstimate(disk_download_gb=1.0, disk_cached_gb=100.0, disk_dump_gb=0.5)
        assert e.total_disk_gb == pytest.approx(1.5)


class TestToDictSerialization:
    def test_findings_round_trip_severity_as_string(self) -> None:
        r = PreflightReport()
        r.add("resource", Severity.OVER, "boom")
        d = r.to_dict()
        assert d["headroom"] == "over"
        assert d["is_feasible"] is False
        assert d["findings"] == [
            {"phase": "resource", "severity": "over", "message": "boom", "metric": None},
        ]

    def test_hardware_and_dataset_pass_through(self) -> None:
        r = PreflightReport(
            hardware={"accelerator": "cuda", "vram_gb": 8.0},
            dataset={"n_samples": 100, "n_classes": 5},
        )
        d = r.to_dict()
        assert d["hardware"]["accelerator"] == "cuda"
        assert d["dataset"]["n_samples"] == 100

    def test_finding_is_frozen(self) -> None:
        f = Finding(phase="resource", severity=Severity.AMPLE, message="ok")
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.message = "changed"  # type: ignore[misc]
