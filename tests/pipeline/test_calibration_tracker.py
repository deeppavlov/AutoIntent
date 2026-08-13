"""Tests for the calibration script's _ModuleTracker callback."""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts/ to sys.path so the test can import calibrate_advisor.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from calibrate_advisor import (  # noqa: E402
    _ModuleTracker,
    _StepTimingCallback,
    _classify_module_role,
    _sum_time_by_role,
)


def test_tracker_records_wall_time_per_module() -> None:
    """One (module, num) → one record with a positive duration."""
    tracker = _ModuleTracker()

    tracker.start_module("linear", 0, {"cv": 3})
    tracker.end_module()

    tracker.start_module("catboost", 1, {"iterations": 100, "depth": 6})
    tracker.end_module()

    assert len(tracker.records) == 2
    assert tracker.records[0]["module"] == "linear"
    assert tracker.records[0]["num"] == 0
    assert tracker.records[0]["config"] == {"cv": 3}
    assert tracker.records[0]["duration_s"] >= 0
    assert tracker.records[1]["module"] == "catboost"
    assert tracker.records[1]["config"] == {"iterations": 100, "depth": 6}


def test_tracker_filters_non_scalar_config_values() -> None:
    """Complex objects in module_kwargs must not appear in the recorded config."""
    tracker = _ModuleTracker()
    tracker.start_module(
        "bert",
        0,
        {"cv": 3, "classification_model_config": {"model_name": "microsoft/deberta"}, "flag": True},
    )
    tracker.end_module()
    assert tracker.records[0]["config"] == {"cv": 3, "flag": True}


def test_end_module_without_start_is_noop() -> None:
    """Defensive: no crash when end_module is called without a matching start."""
    tracker = _ModuleTracker()
    tracker.end_module()  # must not raise
    assert tracker.records == []


def test_records_are_json_serialisable() -> None:
    """Records must survive round-tripping through json.dumps for the CalibrationRow output."""
    import json

    tracker = _ModuleTracker()
    tracker.start_module("linear", 0, {"cv": 3, "unused_none": None})
    tracker.end_module()
    payload = json.dumps(tracker.records)
    assert "linear" in payload
    assert "duration_s" in payload


def test_running_peak_survives_low_last_module() -> None:
    """peak_vram_gb_overall must retain the max across all modules.

    Pins the fix for a bug where torch.cuda's per-module reset_peak_memory_stats
    clobbered the top-level VRAM reading with the last (usually CPU-only) module.
    """
    tracker = _ModuleTracker()

    # Simulate a big embedder module. Populate _current directly to sidestep
    # the real torch.cuda call and inject a synthetic peak.
    tracker.start_module("linear", 0, {"cv": 3})
    tracker.end_module()
    tracker.records[-1]["peak_vram_gb"] = 2.5
    tracker.peak_vram_gb_overall = max(tracker.peak_vram_gb_overall, 2.5)

    # Then a decision module that touches no VRAM.
    tracker.start_module("threshold", 1, {"thresh": 0.5})
    tracker.end_module()
    tracker.records[-1]["peak_vram_gb"] = 0.01
    tracker.peak_vram_gb_overall = max(tracker.peak_vram_gb_overall, 0.01)

    assert tracker.peak_vram_gb_overall == 2.5, "running peak clobbered by later small module"


def test_role_classification_and_time_decomposition() -> None:
    """Each record carries a role, and _sum_time_by_role folds durations correctly.

    Pins R4-P1 #30: classic-preset wall-time must be decomposable into
    embedder-forward vs scorer-fit vs decision-search so a consumer can
    validate the advisor's per-role predictions without post-hoc classification.
    """
    assert _classify_module_role("sentence_transformer") == "embedder"
    assert _classify_module_role("hashing_vectorizer") == "embedder"
    assert _classify_module_role("linear") == "scorer"
    assert _classify_module_role("bert") == "scorer"
    assert _classify_module_role("threshold") == "decision"
    assert _classify_module_role("argmax") == "decision"

    tracker = _ModuleTracker()
    tracker.start_module("sentence_transformer", 0, {})
    tracker.end_module()
    tracker.records[-1]["duration_s"] = 8.0
    tracker.start_module("linear", 1, {})
    tracker.end_module()
    tracker.records[-1]["duration_s"] = 2.0
    tracker.start_module("threshold", 2, {})
    tracker.end_module()
    tracker.records[-1]["duration_s"] = 0.5

    assert [r["role"] for r in tracker.records] == ["embedder", "scorer", "decision"]
    totals = _sum_time_by_role(tracker.records)
    assert totals == {"embedder": 8.0, "scorer": 2.0, "decision": 0.5}


def test_step_timing_callback_answers_every_hf_hook() -> None:
    """HF dispatches every hook via bare getattr — callback must be a real
    TrainerCallback subclass so all defaults are inherited as no-ops.
    Regression: AttributeError on on_train_begin during a bert trial."""
    from transformers import TrainerCallback

    cb = _StepTimingCallback(sink=[])
    assert isinstance(cb, TrainerCallback)
    for name in ["on_init_end", "on_train_begin", "on_epoch_begin", "on_log", "on_save", "on_train_end"]:
        hook = getattr(cb, name)
        assert hook(None, None, "control", model=None) is None, name


def test_peak_sampler_records_cuda_current_allocation() -> None:
    """CUDA polling API: sample_cuda=True → 0.0 float, False → None. Thread
    starts and stops cleanly with no CUDA hardware present."""
    from calibrate_advisor import _PeakSampler

    s = _PeakSampler(sample_cuda=True)
    assert s.peak_cuda_gb == 0.0
    assert _PeakSampler(sample_cuda=False).peak_cuda_gb is None
    import time
    with s:
        time.sleep(0.15)
    assert s.peak_cuda_gb == 0.0
