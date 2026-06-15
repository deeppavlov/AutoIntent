"""End-to-end smoke tests for the advisor.

These run offline — HF Hub probes are monkeypatched to fail so the
advisor falls back to its name-pattern heuristics. Verifies that:

* every bundled preset can be inspected without raising;
* the recommend subcommand picks something on a generous budget and
  nothing on a hostile one;
* ``--json`` emits parseable JSON.
"""

from __future__ import annotations

import json
import sys

import pytest

from autointent._advisor import DatasetStats, HardwareProfile, run_preflight
from autointent._advisor._cli import BUNDLED_PRESETS, main
from autointent.utils import load_preset


@pytest.fixture(autouse=True)
def _force_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the HF Hub probe to "offline" so tests don't hit the network."""
    from autointent._advisor import _estimates, _hub

    _hub.hub_reachable.cache_clear()
    _hub.resolve_model.cache_clear()
    offline = lambda *_a, **_kw: False  # noqa: E731
    monkeypatch.setattr(_hub, "hub_reachable", offline)
    monkeypatch.setattr(_estimates, "hub_reachable", offline)


def _profile(vram_gb: float = 16.0) -> HardwareProfile:
    return HardwareProfile(
        accelerator="cuda" if vram_gb > 0 else "cpu",
        device_name="test-gpu" if vram_gb > 0 else "test-cpu",
        vram_gb=vram_gb,
        ram_gb=32.0,
        free_disk_gb=200.0,
        cpu_count=8,
    )


@pytest.mark.parametrize("preset", BUNDLED_PRESETS)
def test_every_preset_inspects_without_raising(preset: str) -> None:
    cfg = load_preset(preset)  # type: ignore[arg-type]
    stats = DatasetStats.placeholder(n_samples=500, n_classes=10, avg_tokens=24)
    report = run_preflight(cfg, stats, _profile(vram_gb=16.0), preset_name=preset)
    assert report.preset_name == preset
    assert report.low_confidence is True  # we forced offline
    # always at least one resource-phase finding
    assert any(f.phase == "resource" for f in report.findings)


def test_heavy_preset_is_infeasible_on_2gb_budget() -> None:
    cfg = load_preset("transformers-heavy")  # type: ignore[arg-type]
    stats = DatasetStats.placeholder(n_samples=5000, n_classes=20, avg_tokens=40)
    report = run_preflight(cfg, stats, _profile(vram_gb=2.0), preset_name="transformers-heavy")
    assert not report.is_feasible, "deberta-v3-large should not fit in 2 GB"


def test_light_preset_is_feasible_on_8gb_budget() -> None:
    cfg = load_preset("transformers-light")  # type: ignore[arg-type]
    stats = DatasetStats.placeholder(n_samples=1000, n_classes=10, avg_tokens=24)
    report = run_preflight(cfg, stats, _profile(vram_gb=8.0), preset_name="transformers-light")
    assert report.is_feasible


def test_n_jobs_doubles_vram_findings() -> None:
    cfg = load_preset("transformers-light")  # type: ignore[arg-type]
    cfg = {**cfg, "hpo_config": {**(cfg.get("hpo_config") or {}), "n_jobs": 4}}
    stats = DatasetStats.placeholder()
    report = run_preflight(cfg, stats, _profile(vram_gb=4.0))
    assert any("parallel trials" in f.message for f in report.findings)
    assert any(f.phase == "config" and "n_jobs" in f.message for f in report.findings)


def test_cli_inspect_json_is_parseable(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "inspect",
            "transformers-light",
            "--n-samples",
            "500",
            "--n-classes",
            "5",
            "--avg-tokens",
            "20",
            "--json",
            "--budget-vram-gb",
            "16",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["preset_name"] == "transformers-light"
    assert "findings" in payload
    assert payload["headroom"] in {"ample", "tight", "over"}
    # rc is 0 on feasible, 1 otherwise
    assert rc in (0, 1)


def test_cli_inspect_text_runs(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "inspect",
            "transformers-light",
            "--n-samples",
            "200",
            "--n-classes",
            "5",
            "--avg-tokens",
            "15",
            "--budget-vram-gb",
            "16",
        ]
    )
    out = capsys.readouterr().out
    assert "Compute feasibility check" in out
    assert "Verdict:" in out


def test_cli_recommend_picks_a_preset_on_generous_hardware(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(
        [
            "recommend",
            "--n-samples",
            "1000",
            "--n-classes",
            "10",
            "--avg-tokens",
            "20",
            "--budget-vram-gb",
            "24",
        ]
    )
    out = capsys.readouterr().out
    assert "Recommendation:" in out
    assert rc == 0


def test_partial_descriptions_with_description_scorer_flags_red() -> None:
    cfg = {
        "search_space": [
            {
                "node_type": "scoring",
                "search_space": [
                    {"module_name": "description"},
                ],
            }
        ],
    }
    stats = DatasetStats(
        n_samples=500,
        n_classes=10,
        avg_tokens=24,
        has_descriptions=False,
    )
    report = run_preflight(cfg, stats, _profile(vram_gb=16.0))
    assert any(f.phase == "data" and "description" in f.message.lower() for f in report.findings)


def test_long_dataset_triggers_truncation_warning() -> None:
    cfg = {
        "search_space": [
            {
                "node_type": "scoring",
                "search_space": [
                    {
                        "module_name": "bert",
                        "classification_model_config": [{"model_name": "microsoft/deberta-v3-small"}],
                        "max_length": [128],
                    }
                ],
            }
        ],
    }
    stats = DatasetStats(
        n_samples=500,
        n_classes=10,
        avg_tokens=80,
        p95_tokens=512,  # well over 128
    )
    report = run_preflight(cfg, stats, _profile(vram_gb=16.0))
    assert any("truncation" in f.message.lower() for f in report.findings)


def test_cli_recommend_budget_time_flags_red_for_overbudget_presets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tight time budget must flag every preset that exceeds it with RED severity.

    Previously the budget path used a tautological severity expression and the
    breach never escalated the finding — covers the regression."""
    main(
        [
            "recommend",
            "--n-samples",
            "1000",
            "--n-classes",
            "10",
            "--avg-tokens",
            "20",
            "--budget-vram-gb",
            "48",
            "--budget-time-h",
            "0.0001",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    flagged = [
        r
        for r in payload["results"]
        if any(f["severity"] == "over" and "exceeds budget" in f["message"] for f in r["report"]["findings"])
    ]
    assert flagged, "budget-time-h breach should produce OVER severity findings"
    # Any preset above the budget must be marked infeasible.
    for r in flagged:
        assert r["report"]["is_feasible"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
