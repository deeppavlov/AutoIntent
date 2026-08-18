"""Tests for ``autointent.advisor.reduce_to_fit``.

Covers the three review-mandated contracts:

* a feasible config passes through unchanged;
* an infeasible config gets pruned to a config the advisor calls feasible;
* when nothing fits, we raise :class:`ReduceToFitError` — no silent degradation.

Runs fully offline: the same ``_force_offline`` fixture pattern as the sibling
smoke tests, so HF Hub probes fall back to the heuristic large-model shape.
"""

from __future__ import annotations

from typing import Any

import pytest

from autointent.advisor import (
    DatasetStats,
    HardwareProfile,
    ReduceToFitError,
    reduce_to_fit,
    run_preflight,
)


@pytest.fixture(autouse=True)
def _force_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    from autointent.advisor import _hub

    _hub.resolve_model.cache_clear()
    monkeypatch.setattr(_hub, "_hub_metadata", lambda _name: None)


def _profile(vram_gb: float = 16.0, ram_gb: float = 32.0, free_disk_gb: float = 200.0) -> HardwareProfile:
    return HardwareProfile(
        accelerator="cuda" if vram_gb > 0 else "cpu",
        device_name="test-gpu" if vram_gb > 0 else "test-cpu",
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        free_disk_gb=free_disk_gb,
        cpu_count=8,
    )


def _cheap_config() -> dict[str, Any]:
    return {
        "search_space": [
            {
                "node_type": "scoring",
                "target_metric": "scoring_f1",
                "search_space": [{"module_name": "linear"}],
            },
            {
                "node_type": "decision",
                "target_metric": "decision_accuracy",
                "search_space": [{"module_name": "argmax"}],
            },
        ],
    }


def _big_and_cheap_config() -> dict[str, Any]:
    """One expensive transformer + one cheap classic scorer.

    On a tiny (1 GB) VRAM budget, the transformer trips OVER; ``reduce_to_fit``
    should drop it and leave the classic one behind.
    """
    return {
        "search_space": [
            {
                "node_type": "scoring",
                "target_metric": "scoring_f1",
                "search_space": [
                    {
                        "module_name": "bert",
                        "classification_model_config": [{"model_name": "microsoft/deberta-v3-large"}],
                        "batch_size": [128],
                        "max_length": [256],
                    },
                    {"module_name": "linear"},
                ],
            },
            {
                "node_type": "decision",
                "target_metric": "decision_accuracy",
                "search_space": [{"module_name": "argmax"}],
            },
        ],
    }


def _unfittable_config() -> dict[str, Any]:
    return {
        "search_space": [
            {
                "node_type": "scoring",
                "target_metric": "scoring_f1",
                "search_space": [
                    {
                        "module_name": "bert",
                        "classification_model_config": [{"model_name": "microsoft/deberta-v3-large"}],
                        "batch_size": [128],
                        "max_length": [512],
                    },
                ],
            },
            {
                "node_type": "decision",
                "target_metric": "decision_accuracy",
                "search_space": [{"module_name": "argmax"}],
            },
        ],
    }


def test_feasible_config_returns_unchanged() -> None:
    stats = DatasetStats.placeholder(n_samples=500, n_classes=10, avg_tokens=24)
    config = _cheap_config()
    pruned, report = reduce_to_fit(config, stats, _profile(vram_gb=16.0))
    assert report.is_feasible
    # Passthrough: same module still present.
    modules = [e["module_name"] for node in pruned["search_space"] for e in node["search_space"]]
    assert "linear" in modules
    assert "argmax" in modules


def test_prunes_infeasible_transformer_to_classic() -> None:
    stats = DatasetStats.placeholder(n_samples=2000, n_classes=20, avg_tokens=48)
    config = _big_and_cheap_config()

    # Sanity check: base config must be infeasible on a tiny budget, otherwise
    # this test isn't exercising the prune path.
    base = run_preflight(config, stats, _profile(vram_gb=1.0))
    assert not base.is_feasible

    pruned, report = reduce_to_fit(config, stats, _profile(vram_gb=1.0))
    assert report.is_feasible
    modules = [e["module_name"] for node in pruned["search_space"] for e in node["search_space"]]
    assert "bert" not in modules, "expensive transformer should have been dropped"
    assert "linear" in modules, "cheap classic scorer should be preserved"


def test_raises_when_nothing_fits() -> None:
    stats = DatasetStats.placeholder(n_samples=2000, n_classes=20, avg_tokens=48)
    config = _unfittable_config()

    with pytest.raises(ReduceToFitError) as exc_info:
        reduce_to_fit(config, stats, _profile(vram_gb=0.5))

    # The exception carries the final pruned config + last report so callers
    # can inspect what was tried — contract from the review's follow-up.
    err = exc_info.value
    assert err.pruned_config is not None
    assert err.last_report is not None
    # After pruning the only scoring module, the config's scoring node should
    # be gone entirely (or empty), leaving an unfittable pipeline.
    scoring_nodes = [n for n in err.pruned_config["search_space"] if n.get("node_type") == "scoring"]
    assert scoring_nodes == [] or all(not n.get("search_space") for n in scoring_nodes)
