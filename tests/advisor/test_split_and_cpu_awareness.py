"""Tests for the review findings on PR #291: split awareness, cv, and CPU cores.

Three separate complaints, all of which reduced to the advisor ignoring
something it was already being handed:

* the ``cv`` a linear entry actually declares (the time estimate hardcoded 3),
* the fact that the pipeline splits the train split again before any module
  sees it, so raw per-class counts are optimistic,
* ``HardwareProfile.cpu_count``, which was detected and then read by nothing.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from autointent.advisor._estimates._formulas import (
    _LINEAR_PARALLEL_FRACTION,
    _MAX_CPU_SPEEDUP,
    _cores_per_trial,
    _cpu_speedup,
    _logreg_cv_multiplier,
    _time_for_catboost,
    _time_for_linear,
)
from autointent.advisor._hardware import HardwareProfile
from autointent.advisor._report import DatasetStats, PreflightReport, Severity
from autointent.advisor._runner import _data_phase, _effective_train_fraction
from autointent.configs import DataConfig
from autointent.context.data_handler._readiness_util import _min_samples_per_class_for_config


def _stats(class_counts: dict[str, int], *, multilabel: bool = False) -> DatasetStats:
    return DatasetStats(
        n_samples=sum(class_counts.values()),
        n_classes=len(class_counts),
        avg_tokens=16,
        p95_tokens=32,
        multilabel=multilabel,
        class_counts=class_counts,
        source="test",
    )


def _linear_space(cv: int | None = None) -> list[dict[str, Any]]:
    entry: dict[str, Any] = {"module_name": "linear"}
    if cv is not None:
        entry["cv"] = cv
    return [{"node_type": "scoring", "search_space": [entry]}]


def _run_data_phase(stats: DatasetStats, data_config: DataConfig, cv: int | None = None) -> PreflightReport:
    report = PreflightReport()
    _data_phase(_linear_space(cv), stats, data_config, report)
    return report


def _messages(report: PreflightReport) -> str:
    return " | ".join(f.message for f in report.findings)


class TestLogregCvMultiplier:
    """The time estimate used to hardcode 31 = Cs(10) x cv(3) + 1 refit."""

    def test_default_cv_reproduces_the_old_constant(self) -> None:
        assert _logreg_cv_multiplier(3) == 31

    @pytest.mark.parametrize(("cv", "expected"), [(2, 21), (5, 51), (10, 101)])
    def test_scales_with_configured_cv(self, cv: int, expected: int) -> None:
        assert _logreg_cv_multiplier(cv) == expected

    def test_time_grows_with_cv(self) -> None:
        kwargs = {
            "n_trials": 5,
            "n_samples": 10_000,
            "embedder_dim": 768,
            "max_iter": 100,
            "class_multiplier": 20,
        }
        cheap = _time_for_linear(cv_multiplier=_logreg_cv_multiplier(3), **kwargs)
        dear = _time_for_linear(cv_multiplier=_logreg_cv_multiplier(10), **kwargs)
        # cv=10 costs 101/31 as many fits as cv=3; previously both priced the same.
        assert dear == pytest.approx(cheap * 101 / 31)


class TestCpuSpeedup:
    def test_single_core_is_a_no_op(self) -> None:
        assert _cpu_speedup(1, 0.9) == 1.0

    def test_speedup_is_sublinear(self) -> None:
        # Amdahl with p=0.9 on 8 cores is ~4.7x, never the naive 8x.
        assert 1.0 < _cpu_speedup(8, 0.9) < 8.0

    def test_capped_however_many_cores(self) -> None:
        assert _cpu_speedup(1024, 0.99) == _MAX_CPU_SPEEDUP

    def test_never_optimistic_past_the_cap(self) -> None:
        assert _cpu_speedup(10_000, 1.0) <= _MAX_CPU_SPEEDUP

    @pytest.mark.parametrize(("cpu_count", "n_jobs", "expected"), [(16, 1, 16), (16, 4, 4), (16, 32, 1), (0, 1, 1)])
    def test_cores_per_trial_divides_by_concurrent_trials(self, cpu_count: int, n_jobs: int, expected: int) -> None:
        assert _cores_per_trial(cpu_count, n_jobs) == expected


class TestCpuCountReachesTimeEstimates:
    """The complaint was that core count changed nothing. It must now change something."""

    _CATBOOST: ClassVar[dict[str, int]] = {
        "n_trials": 3,
        "n_samples": 10_000,
        "n_features": 768,
        "iterations": 1000,
        "depth": 6,
        "class_multiplier": 10,
    }

    def test_catboost_cpu_time_falls_with_cores(self) -> None:
        one = _time_for_catboost(on_gpu=False, cores=1, **self._CATBOOST)
        many = _time_for_catboost(on_gpu=False, cores=16, **self._CATBOOST)
        assert many < one

    def test_catboost_gpu_time_ignores_cores(self) -> None:
        one = _time_for_catboost(on_gpu=True, cores=1, **self._CATBOOST)
        many = _time_for_catboost(on_gpu=True, cores=64, **self._CATBOOST)
        assert one == many

    def test_linear_time_falls_with_cores_but_less_than_catboost(self) -> None:
        kwargs = {
            "n_trials": 5,
            "n_samples": 10_000,
            "embedder_dim": 768,
            "max_iter": 100,
            "cv_multiplier": 31,
            "class_multiplier": 20,
        }
        one = _time_for_linear(cores=1, **kwargs)
        many = _time_for_linear(cores=16, **kwargs)
        assert many < one
        # L-BFGS only threads inside BLAS, so it must not claim CatBoost's speedup.
        assert one / many == pytest.approx(_cpu_speedup(16, _LINEAR_PARALLEL_FRACTION))

    def test_cpu_count_is_wired_through_run_preflight(self) -> None:
        """End to end: two identical configs differing only in cpu_count must differ in time."""
        from autointent.advisor import run_preflight

        config = {
            "search_space": [
                {"node_type": "scoring", "search_space": [{"module_name": "catboost", "iterations": 1000}]}
            ]
        }
        stats = DatasetStats(n_samples=10_000, n_classes=20, avg_tokens=16, source="test")

        def profile(cpu_count: int) -> HardwareProfile:
            return HardwareProfile(
                accelerator="cpu",
                device_name="test-cpu",
                vram_gb=0.0,
                ram_gb=32.0,
                free_disk_gb=200.0,
                cpu_count=cpu_count,
            )

        small = run_preflight(config, stats, profile(1))
        big = run_preflight(config, stats, profile(32))
        assert big.resource.time_hours < small.resource.time_hours


class TestEffectiveTrainFraction:
    def test_holdout_removes_the_validation_share(self) -> None:
        assert _effective_train_fraction(DataConfig(validation_size=0.2)) == pytest.approx(0.8)

    def test_cross_validation_leaves_one_fold_out(self) -> None:
        assert _effective_train_fraction(DataConfig(scheme="cv", n_folds=5)) == pytest.approx(0.8)

    def test_separation_ratio_shrinks_it_further(self) -> None:
        cfg = DataConfig(validation_size=0.2, separation_ratio=0.5)
        assert _effective_train_fraction(cfg) == pytest.approx(0.4)

    def test_never_leaves_the_unit_interval(self) -> None:
        assert 0.0 <= _effective_train_fraction(DataConfig(validation_size=1.0)) <= 1.0


class TestSplitReadinessAgreement:
    """The advisor must not green-light a dataset the splitter would reject."""

    @pytest.mark.parametrize(
        "data_config",
        [
            DataConfig(),
            DataConfig(scheme="cv", n_folds=5),
            DataConfig(separation_ratio=0.5),
            DataConfig(scheme="cv", n_folds=10, separation_ratio=0.3),
        ],
    )
    def test_advisor_flags_exactly_what_the_splitter_rejects(self, data_config: DataConfig) -> None:
        """Pins the advisor to `check_split_readiness`'s minimum, so the two cannot drift apart."""
        minimum = _min_samples_per_class_for_config(config=data_config)
        # One class one sample below the splitter's own threshold.
        stats = _stats({"ok": 500, "starved": minimum - 1})
        report = _run_data_phase(stats, data_config)
        assert "Stratified splitting will fail" in _messages(report)
        assert "starved" in _messages(report)
        assert any(f.severity is Severity.OVER for f in report.findings)

    @pytest.mark.parametrize(
        "data_config",
        [DataConfig(), DataConfig(scheme="cv", n_folds=5), DataConfig(separation_ratio=0.5)],
    )
    def test_silent_when_every_class_clears_the_threshold(self, data_config: DataConfig) -> None:
        minimum = _min_samples_per_class_for_config(config=data_config)
        stats = _stats({"a": minimum * 100, "b": minimum * 100})
        report = _run_data_phase(stats, data_config)
        assert "Stratified splitting will fail" not in _messages(report)


class TestLogregCheckAccountsForTheSplit:
    def test_class_that_only_passes_on_the_raw_split_is_flagged(self) -> None:
        """The regression: 4 samples clears cv=3 before splitting, and fails after."""
        stats = _stats({"plenty": 500, "borderline": 4})
        cfg = DataConfig(validation_size=0.2)  # 4 * 0.8 = 3.2 -> 3 usable... still >= 3
        assert int(4 * _effective_train_fraction(cfg)) == 3

        # With separation_ratio the same class drops to 4 * 0.8 * 0.5 = 1 usable sample.
        split_cfg = DataConfig(validation_size=0.2, separation_ratio=0.5)
        report = _run_data_phase(stats, split_cfg, cv=3)
        assert "LogisticRegressionCV (cv=3) will fail" in _messages(report)
        assert "borderline" in _messages(report)

    def test_message_names_the_split_when_one_applies(self) -> None:
        stats = _stats({"plenty": 500, "thin": 3})
        report = _run_data_phase(stats, DataConfig(validation_size=0.2), cv=3)
        assert "after the 80% train/validation split" in _messages(report)

    def test_generous_class_counts_stay_silent(self) -> None:
        stats = _stats({"a": 1000, "b": 1000})
        report = _run_data_phase(stats, DataConfig(validation_size=0.2), cv=3)
        assert "LogisticRegressionCV" not in _messages(report)

    def test_multilabel_skips_the_cv_check(self) -> None:
        """Multilabel uses plain LogisticRegression, which has no inner CV."""
        stats = _stats({"a": 1000, "thin": 1}, multilabel=True)
        report = _run_data_phase(stats, DataConfig(validation_size=0.2), cv=3)
        assert "LogisticRegressionCV" not in _messages(report)

    def test_declared_cv_is_used_not_the_default(self) -> None:
        stats = _stats({"a": 1000, "mid": 40})
        assert "LogisticRegressionCV" not in _messages(_run_data_phase(stats, DataConfig(), cv=3))
        assert "LogisticRegressionCV (cv=50) will fail" in _messages(_run_data_phase(stats, DataConfig(), cv=50))
