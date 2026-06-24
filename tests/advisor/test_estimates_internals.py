"""Targeted tests for `_estimates` helpers + edge cases of `run_preflight`."""

from __future__ import annotations

from typing import Any

import pytest

from autointent._advisor import _estimates, _hub, run_preflight
from autointent._advisor._estimates._formulas import _classify_severity, _ram_for_module, _vram_for_transformer
from autointent._advisor._estimates._search_space import _extract_model_names, _max_int
from autointent._advisor._hardware import HardwareProfile
from autointent._advisor._hub import ModelMeta
from autointent._advisor._report import DatasetStats, Severity

# Per-name ModelMeta fixtures used by the offline tests. Production resolution
# (HF Hub config.json + safetensors metadata) is mocked away so the batch-fit
# math doesn't depend on whatever fallback the heuristic path returns.
_FAKE_SHAPES: dict[str, tuple[int, int, int]] = {
    # (total_params, hidden_size, n_layers)
    "microsoft/deberta-v3-large": (350_000_000, 1024, 24),
    "microsoft/deberta-v3-small": (140_000_000, 768, 6),
    "sentence-transformers/all-MiniLM-L6-v2": (33_000_000, 384, 6),
    "intfloat/multilingual-e5-large-instruct": (560_000_000, 1024, 24),
}


def _fake_resolve(model_name: str) -> ModelMeta:
    known = _FAKE_SHAPES.get(model_name)
    params, hidden, layers = known or (110_000_000, 768, 12)
    return ModelMeta(
        name=model_name,
        total_params=params,
        weight_bytes_per_param=4,
        total_file_bytes=params * 4,
        cached_locally=False,
        confidence="hub" if known else "heuristic",
        hidden_size=hidden,
        n_layers=layers,
    )


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _hub.resolve_model.cache_clear()
    monkeypatch.setattr(_hub, "_is_warm_cached", lambda _name: False)
    # Inject deterministic ModelMeta per name; both the _hub re-export and the
    # _estimates rebinding need to be replaced for run_preflight to pick it up.
    monkeypatch.setattr(_hub, "resolve_model", _fake_resolve)
    monkeypatch.setattr(_estimates, "resolve_model", _fake_resolve)


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
        assert _classify_severity(estimate=1.0, budget=10.0) == Severity.AMPLE

    def test_above_yellow_threshold(self) -> None:
        assert _classify_severity(estimate=9.5, budget=10.0) == Severity.TIGHT

    def test_at_or_above_red_threshold(self) -> None:
        assert _classify_severity(estimate=10.0, budget=10.0) == Severity.OVER
        assert _classify_severity(estimate=12.0, budget=10.0) == Severity.OVER

    def test_zero_budget_returns_yellow(self) -> None:
        assert _classify_severity(estimate=1.0, budget=0.0) == Severity.TIGHT


class TestVramForTransformer:
    @pytest.fixture
    def meta(self) -> ModelMeta:
        return ModelMeta(
            name="x",
            total_params=100_000_000,
            weight_bytes_per_param=4,
            total_file_bytes=0,
            cached_locally=False,
            confidence="hub",
        )

    def test_full_finetune_is_larger_than_lora_is_larger_than_inference(self, meta: ModelMeta) -> None:
        inference = _vram_for_transformer(meta, "inference")
        lora = _vram_for_transformer(meta, "lora")
        full = _vram_for_transformer(meta, "full-finetune")
        assert inference < lora < full

    def test_inference_activations_are_smaller_than_training(self, meta: ModelMeta) -> None:
        """Inference doesn't store per-layer outputs for backward — activation memory
        should be many times smaller than training at the same batch_size."""
        train_total = _vram_for_transformer(meta, "full-finetune", batch_size=64, seq_len=128)
        train_weights = _vram_for_transformer(meta, "full-finetune", batch_size=0)
        inf_total = _vram_for_transformer(meta, "inference", batch_size=64, seq_len=128)
        inf_weights = _vram_for_transformer(meta, "inference", batch_size=0)
        train_acts = train_total - train_weights
        inf_acts = inf_total - inf_weights
        assert inf_acts > 0
        assert train_acts > inf_acts
        # 12-layer model: training activations should be at least ~5x inference.
        assert train_acts / inf_acts > 5


def test_ram_scales_with_dataset_size() -> None:
    meta = ModelMeta(
        name="x",
        total_params=100_000_000,
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
                            "classification_model_config": [{"model_name": "microsoft/deberta-v3-small"}],
                            "num_train_epochs": [3],
                            "batch_size": [16],
                        }
                    ],
                }
            ],
            "hpo_config": {"n_trials": 5},
            "logging_config": {"dump_modules": True},
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
                            "classification_model_config": [{"model_name": "microsoft/deberta-v3-small"}],
                            "num_train_epochs": [3],
                            "batch_size": [16],
                        }
                    ],
                }
            ],
            "hpo_config": {"n_trials": 10},
        }
        baseline = run_preflight(cfg, DatasetStats.placeholder(), _profile())
        bumped = run_preflight(cfg, DatasetStats.placeholder(), _profile(), refit_after=True)
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
        assert any(f.phase == "config" and "CatBoost" in f.message for f in report.findings)

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
        assert not any(f.phase == "config" and "CatBoost" in f.message for f in report.findings)

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
        assert any("Heuristic fallback" in n for n in report.notes)

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
            f.phase == "data" and "LogisticRegressionCV" in f.message and f.severity == Severity.OVER
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
                            "classification_model_config": [{"model_name": "some/model"}],
                        }
                    ],
                }
            ]
        }
        stats = DatasetStats(n_samples=500, n_classes=5, avg_tokens=50, p95_tokens=400)
        report = run_preflight(cfg, stats, _profile())
        red = [f for f in report.findings if f.phase == "data" and f.severity == Severity.OVER]
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
                            "classification_model_config": [{"model_name": "some/model"}],
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
            if f.phase == "data" and f.severity == Severity.TIGHT and "truncation" in f.message.lower()
        ]
        assert yellows


class TestLinearCatboostFormulas:
    """Cost surfaces for the classic (sklearn / catboost) scorers."""

    def _embedder_node(self) -> dict[str, Any]:
        return {
            "node_type": "embedder",
            "search_space": [
                {
                    "module_name": "sentence_transformer",
                    "embedder_config": [{"model_name": "sentence-transformers/all-MiniLM-L6-v2"}],
                }
            ],
        }

    def test_linear_contributes_ram_and_time(self) -> None:
        cfg = {
            "search_space": [
                self._embedder_node(),
                {
                    "node_type": "scoring",
                    "search_space": [{"module_name": "linear", "max_iter": [200]}],
                },
            ],
            "hpo_config": {"n_trials": 5},
        }
        stats = DatasetStats.placeholder(n_samples=100_000, n_classes=10, avg_tokens=24)
        report = run_preflight(cfg, stats, _profile())
        linear_drivers = [d for d in report.resource.drivers if d["module"] == "linear"]
        assert len(linear_drivers) == 1
        assert report.resource.ram_gb > 0
        assert report.resource.time_hours > 0
        assert linear_drivers[0]["vram_gb"] == 0  # sklearn is CPU-only

    def test_logreg_cv_multiplier_dominates_multiclass_time(self) -> None:
        """Multiclass linear uses LogisticRegressionCV (Cs*cv+1 ≈ 31 inner fits);
        multilabel uses one LogReg per class (cv_multiplier=1). At equal n_classes,
        multiclass must be much slower than the per-class multilabel path."""
        base = {
            "search_space": [
                self._embedder_node(),
                {
                    "node_type": "scoring",
                    "search_space": [{"module_name": "linear", "max_iter": [1000]}],
                },
            ],
            "hpo_config": {"n_trials": 1},
        }
        multiclass = run_preflight(
            base,
            DatasetStats.placeholder(n_samples=100_000, n_classes=10, multilabel=False),
            _profile(),
        )
        multilabel = run_preflight(
            base,
            DatasetStats.placeholder(n_samples=100_000, n_classes=10, multilabel=True),
            _profile(),
        )
        # multiclass: 31 inner fits x 1 model; multilabel: 1 fit x n_classes=10 models.
        # 31 > 10 => multiclass is the slower path.
        assert multiclass.resource.time_hours > multilabel.resource.time_hours

    def test_catboost_contributes_ram_and_time_on_cpu(self) -> None:
        cfg = {
            "search_space": [
                self._embedder_node(),
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "catboost",
                            "iterations": [1000],
                            "depth": [6],
                        }
                    ],
                },
            ],
            "hpo_config": {"n_trials": 3},
        }
        stats = DatasetStats.placeholder(n_samples=100_000, n_classes=8, avg_tokens=24)
        report = run_preflight(cfg, stats, _profile(accelerator="cpu"))
        cb = next(d for d in report.resource.drivers if d["module"] == "catboost")
        assert report.resource.ram_gb > 0
        assert report.resource.time_hours > 0
        assert cb["vram_gb"] == 0
        assert cb["mode"] == "catboost"

    def test_catboost_gpu_moves_cost_to_vram(self) -> None:
        cfg = {
            "search_space": [
                self._embedder_node(),
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "catboost",
                            "iterations": [1000],
                            "depth": [6],
                            "task_type": "GPU",
                        }
                    ],
                },
            ],
            "hpo_config": {"n_trials": 2},
        }
        stats = DatasetStats.placeholder(n_samples=100_000, n_classes=8, avg_tokens=24)
        report = run_preflight(cfg, stats, _profile(accelerator="cuda"))
        cb = next(d for d in report.resource.drivers if d["module"] == "catboost")
        assert report.resource.vram_gb > 0
        assert cb["ram_gb"] == 0
        assert cb["mode"] == "catboost-gpu"

    def test_linear_scales_with_n_samples(self) -> None:
        cfg = {
            "search_space": [
                self._embedder_node(),
                {
                    "node_type": "scoring",
                    "search_space": [{"module_name": "linear"}],
                },
            ],
        }
        small = run_preflight(cfg, DatasetStats.placeholder(n_samples=500), _profile())
        big = run_preflight(cfg, DatasetStats.placeholder(n_samples=500_000), _profile())
        assert big.resource.time_hours > small.resource.time_hours
        assert big.resource.ram_gb > small.resource.ram_gb


class TestPerDriverBatchHint:
    """Each transformer driver carries its own (batch_size, max_batch_size) for rendering."""

    def _bert_cfg(self, model_name: str, batch_size: int) -> dict[str, Any]:
        return {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "bert",
                            "classification_model_config": [{"model_name": model_name}],
                            "num_train_epochs": [3],
                            "batch_size": [batch_size],
                        }
                    ],
                }
            ],
            "hpo_config": {"n_trials": 1},
        }

    def test_driver_records_current_and_max_batch(self) -> None:
        report = run_preflight(
            self._bert_cfg("microsoft/deberta-v3-large", batch_size=64),
            DatasetStats.placeholder(),
            _profile(vram_gb=7.5),
        )
        drivers = [d for d in report.resource.drivers if d["module"] == "bert"]
        assert drivers
        d = drivers[0]
        assert d["batch_size"] == 64
        # vram_gb=7.5 against ~5.9 GB weights x 0.9 tight ratio -> little activation room, max < 64.
        assert d["max_batch_size"] is not None
        assert 0 < d["max_batch_size"] < 64

    def test_max_batch_zero_when_weights_alone_overflow(self) -> None:
        report = run_preflight(
            self._bert_cfg("microsoft/deberta-v3-large", batch_size=64),
            DatasetStats.placeholder(),
            _profile(vram_gb=2.0),
        )
        d = next(d for d in report.resource.drivers if d["module"] == "bert")
        assert d["max_batch_size"] == 0

    def test_max_batch_can_be_larger_than_current(self) -> None:
        report = run_preflight(
            self._bert_cfg("microsoft/deberta-v3-large", batch_size=32),
            DatasetStats.placeholder(),
            _profile(vram_gb=64.0),
        )
        d = next(d for d in report.resource.drivers if d["module"] == "bert")
        assert d["max_batch_size"] is not None
        assert d["max_batch_size"] > 32

    def test_multiple_drivers_carry_independent_max_batch(self) -> None:
        cfg = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "bert",
                            "classification_model_config": [
                                {"model_name": "microsoft/deberta-v3-small"},
                                {"model_name": "microsoft/deberta-v3-large"},
                            ],
                            "num_train_epochs": [3],
                            "batch_size": [64],
                        }
                    ],
                }
            ],
            "hpo_config": {"n_trials": 1},
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile(vram_gb=10.0))
        small = next(d for d in report.resource.drivers if "small" in d["model"])
        large = next(d for d in report.resource.drivers if "large" in d["model"])
        # The smaller model has more headroom -> larger max batch (or equal-cap when both saturate).
        assert small["max_batch_size"] >= large["max_batch_size"]


class TestDumpModulesBounding:
    """`dump_modules=True` writes one selected variant per node per trial — not
    every candidate. The estimate must be bounded by sum-of-max-per-node x n_trials."""

    def test_dump_disk_is_bounded_by_per_node_max_not_sum_of_all_variants(self) -> None:
        # Two BERT candidates in the same node: only one is selected per trial.
        cfg = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "bert",
                            "classification_model_config": [
                                {"model_name": "microsoft/deberta-v3-small"},
                                {"model_name": "microsoft/deberta-v3-large"},
                            ],
                            "num_train_epochs": [3],
                            "batch_size": [16],
                        }
                    ],
                }
            ],
            "hpo_config": {"n_trials": 4},
            "logging_config": {"dump_modules": True},
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile())
        # Per-node max ~ deberta-v3-large weights (~350M x 4 ~ 1.3 GB). Two-candidate
        # sum would be roughly doubled. Verify we used the per-node-max bound.
        small_meta = _hub.resolve_model("microsoft/deberta-v3-small")
        large_meta = _hub.resolve_model("microsoft/deberta-v3-large")
        expected = large_meta.weights_gb * 4
        naive_sum = (small_meta.weights_gb + large_meta.weights_gb) * 4
        assert report.resource.disk_dump_gb == pytest.approx(expected, rel=0.01)
        assert report.resource.disk_dump_gb < naive_sum

    def test_dump_disk_sums_across_nodes(self) -> None:
        cfg = {
            "search_space": [
                {
                    "node_type": "embedder",
                    "search_space": [
                        {
                            "module_name": "sentence_transformer",
                            "embedder_config": [{"model_name": "sentence-transformers/all-MiniLM-L6-v2"}],
                        }
                    ],
                },
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "bert",
                            "classification_model_config": [{"model_name": "microsoft/deberta-v3-small"}],
                            "num_train_epochs": [3],
                            "batch_size": [16],
                        }
                    ],
                },
            ],
            "hpo_config": {"n_trials": 2},
            "logging_config": {"dump_modules": True},
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile())
        embedder = _hub.resolve_model("sentence-transformers/all-MiniLM-L6-v2")
        bert = _hub.resolve_model("microsoft/deberta-v3-small")
        expected = (embedder.weights_gb + bert.weights_gb) * 2
        assert report.resource.disk_dump_gb == pytest.approx(expected, rel=0.01)
