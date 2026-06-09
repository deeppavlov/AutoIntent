"""Targeted tests for `_estimates` helpers + edge cases of `run_preflight`."""

from __future__ import annotations

import pytest

from autointent._advisor import _estimates, _hub
from autointent._advisor._estimates import (
    _classify_severity,
    _extract_model_names,
    _max_int,
    _ram_for_module,
    _vram_for_transformer,
    run_preflight,
)
from autointent._advisor._hardware import HardwareProfile
from autointent._advisor._hub import ModelMeta
from autointent._advisor._report import DatasetStats, Severity


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _hub.hub_reachable.cache_clear()
    _hub.resolve_model.cache_clear()
    offline = lambda *_a, **_kw: False  # noqa: E731
    monkeypatch.setattr(_hub, "hub_reachable", offline)
    monkeypatch.setattr(_estimates, "hub_reachable", offline)
    monkeypatch.setattr(_hub, "_is_warm_cached", lambda _name: False)


def _profile(vram_gb: float = 16.0, accelerator: str = "cuda") -> HardwareProfile:
    return HardwareProfile(
        accelerator=accelerator,  # type: ignore[arg-type]
        device_name=f"test-{accelerator}",
        vram_gb=vram_gb,
        ram_gb=32.0,
        free_disk_gb=200.0,
        cpu_count=8,
    )


class TestMaxInt:
    def test_none_returns_default(self) -> None:
        assert _max_int(None, 7) == 7

    def test_list_picks_max(self) -> None:
        assert _max_int([1, 5, 3], 0) == 5

    def test_range_dict_uses_high(self) -> None:
        assert _max_int({"low": 1, "high": 9}, 0) == 9

    def test_scalar_int_passes_through(self) -> None:
        assert _max_int(42, 0) == 42

    def test_garbage_returns_default(self) -> None:
        assert _max_int("not-a-number", 11) == 11


class TestExtractModelNames:
    def test_classification_model_config_as_list(self) -> None:
        entry = {"classification_model_config": [{"model_name": "foo/bar"}]}
        assert _extract_model_names(entry) == ["foo/bar"]

    def test_classification_model_config_as_dict(self) -> None:
        entry = {"classification_model_config": {"model_name": "foo/bar"}}
        assert _extract_model_names(entry) == ["foo/bar"]

    def test_embedder_config_picked_up(self) -> None:
        entry = {"embedder_config": [{"model_name": "e/b"}]}
        assert _extract_model_names(entry) == ["e/b"]

    def test_multiple_choices_all_returned(self) -> None:
        entry = {
            "classification_model_config": [
                {"model_name": "a/x"},
                {"model_name": "b/y"},
            ]
        }
        assert _extract_model_names(entry) == ["a/x", "b/y"]

    def test_empty_entry(self) -> None:
        assert _extract_model_names({}) == []


class TestClassifySeverity:
    def test_below_yellow_is_green(self) -> None:
        assert _classify_severity(estimate=1.0, budget=10.0) == Severity.GREEN

    def test_above_yellow_threshold(self) -> None:
        assert _classify_severity(estimate=8.0, budget=10.0) == Severity.YELLOW

    def test_at_or_above_red_threshold(self) -> None:
        assert _classify_severity(estimate=10.0, budget=10.0) == Severity.RED
        assert _classify_severity(estimate=12.0, budget=10.0) == Severity.RED

    def test_zero_budget_returns_yellow(self) -> None:
        assert _classify_severity(estimate=1.0, budget=0.0) == Severity.YELLOW


class TestVramForTransformer:
    @pytest.fixture
    def meta(self) -> ModelMeta:
        return ModelMeta(
            name="x",
            params_millions=100.0,
            weight_bytes_per_param=4,
            total_file_bytes=0,
            cached_locally=False,
            confidence="hub",
        )

    def test_full_finetune_is_larger_than_lora_is_larger_than_inference(
        self, meta: ModelMeta
    ) -> None:
        inference = _vram_for_transformer(meta, "inference", mixed_precision=False)
        lora = _vram_for_transformer(meta, "lora", mixed_precision=False)
        full = _vram_for_transformer(meta, "full-finetune", mixed_precision=False)
        assert inference < lora < full

    def test_amp_does_not_naively_halve(self, meta: ModelMeta) -> None:
        """The proposal calls out that AMP doesn't halve total VRAM — fp32 master
        weights and Adam moments don't shrink. Weight-side accounting comes out
        equal to fp32; the only savings (activations) aren't modeled by us."""
        full_fp32 = _vram_for_transformer(meta, "full-finetune", mixed_precision=False)
        full_amp = _vram_for_transformer(meta, "full-finetune", mixed_precision=True)
        assert full_amp / full_fp32 == pytest.approx(1.0)
        assert full_amp / full_fp32 > 0.5  # explicit check vs the naive-halving formula

    def test_reranker_uses_inference_class(self, meta: ModelMeta) -> None:
        inference = _vram_for_transformer(meta, "inference", mixed_precision=False)
        reranker = _vram_for_transformer(meta, "reranker", mixed_precision=False)
        assert reranker > inference


def test_ram_scales_with_dataset_size() -> None:
    meta = ModelMeta(
        name="x",
        params_millions=100.0,
        weight_bytes_per_param=4,
        total_file_bytes=0,
        cached_locally=False,
        confidence="hub",
    )
    small = _ram_for_module(meta, DatasetStats.placeholder(n_samples=100))
    big = _ram_for_module(meta, DatasetStats.placeholder(n_samples=10_000_000, avg_tokens=128))
    assert big > small


class TestRunPreflightFeatures:
    def test_dump_modules_adds_disk_during_training(self) -> None:
        cfg = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "bert",
                            "classification_model_config": [
                                {"model_name": "microsoft/deberta-v3-small"}
                            ],
                            "num_train_epochs": [3],
                            "batch_size": [16],
                        }
                    ],
                }
            ],
            "hpo_config": {"n_trials": 5},
            "dump_modules": True,
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile())
        assert report.resource.disk_dump_gb > 0
        assert any("during training" in f.message for f in report.findings)

    def test_refit_after_increases_time(self) -> None:
        cfg = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "bert",
                            "classification_model_config": [
                                {"model_name": "microsoft/deberta-v3-small"}
                            ],
                            "num_train_epochs": [3],
                            "batch_size": [16],
                        }
                    ],
                }
            ],
            "hpo_config": {"n_trials": 10},
        }
        baseline = run_preflight(cfg, DatasetStats.placeholder(), _profile())
        cfg_refit = {**cfg, "refit_after": True}
        bumped = run_preflight(cfg_refit, DatasetStats.placeholder(), _profile())
        assert bumped.resource.time_hours > baseline.resource.time_hours

    def test_catboost_gpu_without_cuda_flags_config(self) -> None:
        cfg = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {"module_name": "catboost", "task_type": "GPU"},
                    ],
                }
            ],
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile(accelerator="cpu"))
        assert any(
            f.phase == "config" and "CatBoost" in f.message for f in report.findings
        )

    def test_catboost_gpu_with_cuda_is_silent(self) -> None:
        cfg = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {"module_name": "catboost", "task_type": "GPU"},
                    ],
                }
            ],
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile(accelerator="cuda"))
        assert not any(
            f.phase == "config" and "CatBoost" in f.message for f in report.findings
        )

    def test_offline_flips_low_confidence(self) -> None:
        cfg = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "bert",
                            "classification_model_config": [{"model_name": "any/model"}],
                        }
                    ],
                }
            ]
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile())
        assert report.low_confidence is True
        assert any("HF Hub unreachable" in n for n in report.notes)

    def test_rare_classes_with_linear_scorer_flag_red(self) -> None:
        cfg = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {"module_name": "linear"},
                    ],
                }
            ]
        }
        stats = DatasetStats(
            n_samples=20,
            n_classes=5,
            avg_tokens=10,
            rare_classes=["intent_a", "intent_b"],
        )
        report = run_preflight(cfg, stats, _profile())
        assert any(
            f.phase == "data" and "LogisticRegressionCV" in f.message and f.severity == Severity.RED
            for f in report.findings
        )

    def test_truncation_red_when_p95_dominates_max_length(self) -> None:
        cfg = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "bert",
                            "max_length": [128],
                            "classification_model_config": [
                                {"model_name": "some/model"}
                            ],
                        }
                    ],
                }
            ]
        }
        stats = DatasetStats(n_samples=500, n_classes=5, avg_tokens=50, p95_tokens=400)
        report = run_preflight(cfg, stats, _profile())
        red = [f for f in report.findings if f.phase == "data" and f.severity == Severity.RED]
        assert red, "p95=400 > 1.5 * max_length=128 should be red"

    def test_truncation_yellow_when_p95_only_slightly_exceeds(self) -> None:
        cfg = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "bert",
                            "max_length": [128],
                            "classification_model_config": [
                                {"model_name": "some/model"}
                            ],
                        }
                    ],
                }
            ]
        }
        stats = DatasetStats(n_samples=500, n_classes=5, avg_tokens=50, p95_tokens=140)
        report = run_preflight(cfg, stats, _profile())
        yellows = [
            f
            for f in report.findings
            if f.phase == "data"
            and f.severity == Severity.YELLOW
            and "truncation" in f.message.lower()
        ]
        assert yellows
