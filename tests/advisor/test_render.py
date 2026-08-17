"""Output rendering: text formatting and JSON serialization."""

from __future__ import annotations

import json

from autointent.advisor._render import _batch_hint, render_json, render_recommendation, render_text
from autointent.advisor._report import (
    DatasetStats,
    PreflightReport,
    ResourceEstimate,
    Severity,
)


def _populated_report() -> PreflightReport:
    r = PreflightReport(
        preset_name="example",
        hardware={
            "accelerator": "cuda",
            "device_name": "RTX 3060",
            "vram_gb": 8.0,
            "ram_gb": 32.0,
            "free_disk_gb": 100.0,
            "device_class": "low-gpu",
        },
        dataset={"n_samples": 500, "n_classes": 10, "avg_tokens": 30, "source": "placeholder"},
        resource=ResourceEstimate(
            disk_download_gb=2.5,
            disk_cached_gb=0.5,
            ram_gb=1.0,
            vram_gb=4.0,
            time_hours=1.2,
            drivers=[
                {
                    "node_type": "scoring",
                    "module": "bert",
                    "model": "x/y",
                    "mode": "full-finetune",
                    "vram_gb": 4.0,
                    "ram_gb": 1.0,
                    "time_hours": 1.2,
                    "confidence": "hub",
                }
            ],
        ),
        notes=["MPS unified memory note"],
    )
    r.add("resource", Severity.TIGHT, "VRAM ~6 GB vs available 8 GB")
    r.add("data", Severity.OVER, "rare classes blocked")
    return r


class TestRenderText:
    def test_contains_phase_blocks(self) -> None:
        out = render_text(_populated_report())
        assert "Resource:" in out
        assert "Data:" in out
        # Config phase has no findings -> block omitted
        assert "Config:" not in out

    def test_includes_drivers_block(self) -> None:
        out = render_text(_populated_report())
        assert "Drivers of cost:" in out
        assert "x/y" in out

    def test_verdict_reflects_headroom(self) -> None:
        out = render_text(_populated_report())
        assert "Verdict: INFEASIBLE" in out
        assert "headroom: over" in out

    def test_disclaimer_always_present(self) -> None:
        out = render_text(_populated_report())
        assert "heuristic upper bounds" in out

    def test_low_confidence_tag_when_offline(self) -> None:
        r = _populated_report()
        r.low_confidence = True
        out = render_text(r)
        assert "low-confidence" in out

    def test_preset_name_in_title(self) -> None:
        out = render_text(_populated_report())
        assert "Compute feasibility check — example" in out

    def test_empty_report_still_renders(self) -> None:
        out = render_text(PreflightReport())
        assert "Compute feasibility check" in out
        assert "Verdict: feasible" in out


class TestRenderJson:
    def test_is_valid_json(self) -> None:
        json.loads(render_json(_populated_report()))

    def test_findings_have_string_severity(self) -> None:
        d = json.loads(render_json(_populated_report()))
        for f in d["findings"]:
            assert f["severity"] in {"ample", "tight", "over"}

    def test_headroom_and_feasibility_serialized(self) -> None:
        d = json.loads(render_json(_populated_report()))
        assert d["headroom"] == "over"
        assert d["is_feasible"] is False

    def test_empty_report_serializes(self) -> None:
        d = json.loads(render_json(PreflightReport()))
        assert d["headroom"] == "ample"
        assert d["is_feasible"] is True


class TestRenderRecommendation:
    def _two_reports(self) -> list[tuple[str, PreflightReport]]:
        a = PreflightReport(preset_name="a", resource=ResourceEstimate(vram_gb=2.0, time_hours=0.5))
        a.add("resource", Severity.AMPLE, "ok")
        b = PreflightReport(preset_name="b", resource=ResourceEstimate(vram_gb=8.0, time_hours=4.0))
        b.add("resource", Severity.OVER, "too big")
        return [("a", a), ("b", b)]

    def test_lists_chosen_preset_when_present(self) -> None:
        out = render_recommendation(self._two_reports(), chosen="a")
        assert "-> a" in out

    def test_handles_no_chosen(self) -> None:
        out = render_recommendation(self._two_reports(), chosen=None)
        assert "none of the bundled presets" in out

    def test_includes_all_presets_in_table(self) -> None:
        out = render_recommendation(self._two_reports(), chosen="a")
        assert "a " in out  # preset name
        assert "b " in out

    def test_shows_status_per_preset(self) -> None:
        out = render_recommendation(self._two_reports(), chosen="a")
        assert "feasible" in out
        assert "infeasible" in out


class TestBatchHint:
    """Per-driver batch cell rendered in the Drivers-of-cost table."""

    def test_arrow_when_max_differs(self) -> None:
        assert _batch_hint({"batch_size": 64, "max_batch_size": 32}) == "64 -> 32"

    def test_plain_when_max_equals_current(self) -> None:
        assert _batch_hint({"batch_size": 64, "max_batch_size": 64}) == "64"

    def test_no_fit_label_when_max_zero(self) -> None:
        assert _batch_hint({"batch_size": 64, "max_batch_size": 0}) == "64 (no fit)"

    def test_empty_when_no_batch(self) -> None:
        assert _batch_hint({"batch_size": None, "max_batch_size": None}) == ""

    def test_increase_arrow(self) -> None:
        assert _batch_hint({"batch_size": 32, "max_batch_size": 128}) == "32 -> 128"


def test_dataset_stats_in_text_block() -> None:
    stats = DatasetStats.placeholder(n_samples=777, n_classes=4)
    r = PreflightReport(
        dataset={
            "n_samples": stats.n_samples,
            "n_classes": stats.n_classes,
            "avg_tokens": stats.avg_tokens,
            "source": stats.source,
        }
    )
    out = render_text(r)
    assert "777" in out
    assert "n_classes=4" in out
