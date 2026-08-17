"""Targeted tests for `_estimates` helpers + edge cases of `run_preflight`."""

from __future__ import annotations

from typing import Any

import pytest

from autointent.advisor import _hub, run_preflight
from autointent.advisor._estimates._formulas import _classify_severity, _ram_for_module, _vram_for_transformer
from autointent.advisor._estimates._search_space import _extract_model_names, _max_int
from autointent.advisor._hardware import HardwareProfile
from autointent.advisor._hub import ModelMeta
from autointent.advisor._report import DatasetStats, Severity

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
    # Resource phase calls `_hub.resolve_model(...)` via module reference, so
    # patching the symbol on `_hub` is enough.
    monkeypatch.setattr(_hub, "resolve_model", _fake_resolve)


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
        # Low-confidence used to be a note; it's now a prominent finding so
        # reviewers of the report see it in the main findings block.
        assert any("LOW CONFIDENCE" in f.message for f in report.findings)

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
            class_counts={"intent_a": 1, "intent_b": 2, "intent_c": 6, "intent_d": 6, "intent_e": 5},
        )
        report = run_preflight(cfg, stats, _profile())
        assert any(
            f.phase == "data" and "LogisticRegressionCV (cv=3)" in f.message and f.severity == Severity.OVER
            for f in report.findings
        )

    def test_rare_classes_threshold_follows_entry_cv(self) -> None:
        """When a linear entry sets cv=5, classes with 4 samples should still fail."""
        cfg = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [
                        {"module_name": "linear", "cv": 5},
                    ],
                }
            ]
        }
        # All classes have >=3 samples, so a cv=3 check would pass — but cv=5
        # needs >=5, so intent_a (4 samples) must be flagged.
        stats = DatasetStats(
            n_samples=20,
            n_classes=3,
            avg_tokens=10,
            class_counts={"intent_a": 4, "intent_b": 8, "intent_c": 8},
        )
        report = run_preflight(cfg, stats, _profile())
        assert any(f.phase == "data" and "cv=5" in f.message and "intent_a" in f.message for f in report.findings)

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
        # The "+embed" suffix is added when the embedder forward is folded into
        # this classic entry via the embedding-cache adjustment.
        assert cb["mode"].startswith("catboost")

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
        # The "+embed" suffix is added when the embedder forward is folded into
        # this classic entry via the embedding-cache adjustment.
        assert cb["mode"].startswith("catboost-gpu")

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


class TestEmbeddingCache:
    """Cache-aware time + disk accounting for embedder-honoring scorers.

    autointent's ``SentenceTransformerEmbedding`` (``use_cache=True`` by default)
    persists per-(model, utterances, prompt) embeddings to disk, so subsequent
    trials/modules that reuse the same embedder hit the cache instead of
    re-running the forward pass.
    """

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

    def test_duplicate_knn_entries_zero_time_after_first(self) -> None:
        """Two knn entries sharing an embedder: the second one's forward is free."""
        cfg = {
            "search_space": [
                self._embedder_node(),
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "knn",
                            "embedder_config": [{"model_name": "sentence-transformers/all-MiniLM-L6-v2"}],
                            "batch_size": [32],
                            "max_length": [128],
                        },
                        {
                            "module_name": "knn",
                            "embedder_config": [{"model_name": "sentence-transformers/all-MiniLM-L6-v2"}],
                            "batch_size": [32],
                            "max_length": [128],
                        },
                    ],
                },
            ],
            "hpo_config": {"n_trials": 5},
        }
        # Use a large placeholder so per-step FLOPs are enough to register as
        # non-zero rounded time even for tiny MiniLM. Behavior we're testing is
        # "first entry pays, second is cached" — needs first > 0 to be visible.
        report = run_preflight(cfg, DatasetStats.placeholder(n_samples=1_000_000), _profile())
        knn_drivers = [d for d in report.resource.drivers if d["module"] == "knn"]
        assert len(knn_drivers) == 2
        first, second = knn_drivers
        assert first["time_hours"] > 0
        assert second["time_hours"] == 0
        assert "cached" in second["mode"]

    def test_classic_entry_gets_synthetic_embedder_forward(self) -> None:
        """A linear scorer alone with an embedder: the embedder forward is added once."""
        cfg = {
            "search_space": [
                self._embedder_node(),
                {
                    "node_type": "scoring",
                    "search_space": [{"module_name": "linear"}],
                },
            ],
            "hpo_config": {"n_trials": 3},
        }
        # Re-run with the embedder node removed to compare cleanly.
        cfg_no_embed = {
            "search_space": [
                {
                    "node_type": "scoring",
                    "search_space": [{"module_name": "linear"}],
                },
            ],
            "hpo_config": {"n_trials": 3},
        }
        with_embed = run_preflight(cfg, DatasetStats.placeholder(n_samples=10_000), _profile())
        no_embed = run_preflight(cfg_no_embed, DatasetStats.placeholder(n_samples=10_000), _profile())
        # The linear row gets a "+embed" suffix when an embedder is present.
        linear_with = next(d for d in with_embed.resource.drivers if d["module"] == "linear")
        linear_no = next(d for d in no_embed.resource.drivers if d["module"] == "linear")
        assert "embed" in linear_with["mode"]
        assert linear_with["time_hours"] >= linear_no["time_hours"]

    def test_disk_embedding_cache_scales_with_n_samples(self) -> None:
        """``disk_embedding_cache_gb`` ~ n_samples x hidden_size x 4 bytes per embedder."""
        cfg = {
            "search_space": [
                self._embedder_node(),
                {
                    "node_type": "scoring",
                    "search_space": [{"module_name": "linear"}],
                },
            ],
        }
        small = run_preflight(cfg, DatasetStats.placeholder(n_samples=1_000), _profile())
        big = run_preflight(cfg, DatasetStats.placeholder(n_samples=1_000_000), _profile())
        assert small.resource.disk_embedding_cache_gb > 0
        assert big.resource.disk_embedding_cache_gb > small.resource.disk_embedding_cache_gb * 100

    def test_warm_cache_probe_zeroes_forward_and_disk(self) -> None:
        """When ``embedding_cache_probe`` reports the embedder is warm, the
        advisor must predict 0 forward time AND 0 ``disk_embedding_cache_gb``
        for that model — mirrors HF-weights ``cached_locally`` behavior."""
        cfg = {
            "search_space": [
                self._embedder_node(),
                {
                    "node_type": "scoring",
                    "search_space": [
                        {
                            "module_name": "knn",
                            "embedder_config": [
                                {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
                            ],
                            "batch_size": [32],
                            "max_length": [128],
                        }
                    ],
                },
            ],
            "hpo_config": {"n_trials": 1},
        }
        stats = DatasetStats.placeholder(n_samples=1_000_000)
        cold = run_preflight(cfg, stats, _profile())
        warm = run_preflight(cfg, stats, _profile(), embedding_cache_probe=lambda _name: True)

        cold_knn = next(d for d in cold.resource.drivers if d["module"] == "knn")
        warm_knn = next(d for d in warm.resource.drivers if d["module"] == "knn")

        assert cold_knn["time_hours"] > 0
        assert warm_knn["time_hours"] == 0
        assert "warm" in warm_knn["mode"]
        assert cold.resource.disk_embedding_cache_gb > 0
        # Warm: forward wasn't charged → model isn't in ``cached_embedders`` →
        # no disk_embedding_cache contribution.
        assert warm.resource.disk_embedding_cache_gb == 0


class TestCnnRnnHeuristic:
    """cnn/rnn get a real small-model estimate, not a not-estimated zero row."""

    def test_cnn_row_is_nonzero(self) -> None:
        cfg = {
            "search_space": [
                {"node_type": "scoring", "search_space": [
                    {"module_name": "cnn", "embed_dim": [128], "num_filters": [128],
                     "kernel_sizes": [[3, 4, 5]], "batch_size": [64], "num_train_epochs": [60]},
                ]},
                {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
            ],
            "hpo_config": {"n_trials": 10},
        }
        report = run_preflight(cfg, DatasetStats.placeholder(n_samples=5000, n_classes=20), _profile())
        cnn_row = next(d for d in report.resource.drivers if d["module"] == "cnn")
        # Real numbers (not the not-estimated placeholder)
        assert cnn_row["mode"] == "small-torch-train"
        assert cnn_row["vram_gb"] > 0
        assert cnn_row["ram_gb"] > 0
        assert cnn_row["time_hours"] > 0

    def test_rnn_row_uses_hidden_dim(self) -> None:
        # Bigger hidden_dim → bigger VRAM.
        base = {"module_name": "rnn", "embed_dim": [128], "batch_size": [64], "num_train_epochs": [30]}

        def _run(hidden: int) -> float:
            cfg = {
                "search_space": [
                    {"node_type": "scoring", "search_space": [{**base, "hidden_dim": [hidden]}]},
                    {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
                ],
                "hpo_config": {"n_trials": 5},
            }
            report = run_preflight(cfg, DatasetStats.placeholder(), _profile())
            return float(next(d["vram_gb"] for d in report.resource.drivers if d["module"] == "rnn"))

        assert _run(1024) > _run(128), "larger hidden_dim must produce a larger VRAM row"


class TestNtrialsSharedAcrossVariants:
    """n_trials is a *node* budget shared across module_name candidates."""

    def test_single_module_gets_full_n_trials(self) -> None:
        # Big dataset + embedder so linear time is non-zero and comparable.
        embedder_cfg = {"embedder_config": {"model_name": "intfloat/multilingual-e5-large-instruct"}}
        cfg = {
            **embedder_cfg,
            "search_space": [
                {"node_type": "scoring", "search_space": [
                    {"module_name": "linear"},
                ]},
                {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
            ],
            "hpo_config": {"n_trials": 200},
        }
        cfg2 = {
            **embedder_cfg,
            "search_space": [
                {"node_type": "scoring", "search_space": [
                    {"module_name": "linear"},
                    {"module_name": "knn", "k": [5]},
                ]},
                {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
            ],
            "hpo_config": {"n_trials": 200},
        }
        stats = DatasetStats.placeholder(n_samples=10000, n_classes=77, avg_tokens=24)
        solo = run_preflight(cfg, stats, _profile())
        shared = run_preflight(cfg2, stats, _profile())

        solo_lin = next(d["time_hours"] for d in solo.resource.drivers if d["module"] == "linear")
        shared_lin = next(d["time_hours"] for d in shared.resource.drivers if d["module"] == "linear")
        # Same module, same everything, but shared node has 2 variants → linear
        # sees half the trials.
        assert solo_lin > 0
        assert shared_lin > 0
        assert solo_lin > shared_lin, (
            f"linear alone should get full n_trials, shared should get half; got solo={solo_lin} shared={shared_lin}"
        )
        # Concretely: solo=20 trials, shared=10 trials → 2x ratio (allow slop for rounding).
        ratio = solo_lin / shared_lin
        assert 1.5 < ratio < 2.5, f"expected ~2x ratio, got {ratio}"


class TestProcessBaselineFloor:
    """Every fit reserves ~1.5 GB RAM for torch/transformers/datasets."""

    def test_ram_estimate_never_below_baseline(self) -> None:
        # Minimal preset — no scoring modules that contribute RAM.
        cfg = {
            "search_space": [
                {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
            ],
            "hpo_config": {"n_trials": 1},
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile())
        # The floor is applied as an additive term, so even an empty pipeline
        # must report at least the baseline in RAM.
        assert report.resource.ram_gb >= 1.0

    def test_cuda_vram_baseline_only_when_gpu_used(self) -> None:
        cfg_cpu_only = {
            "search_space": [
                {"node_type": "scoring", "search_space": [{"module_name": "linear"}]},
                {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
            ],
            "hpo_config": {"n_trials": 1},
        }
        # linear scorer runs on CPU only → no CUDA baseline should apply.
        report = run_preflight(cfg_cpu_only, DatasetStats.placeholder(), _profile(accelerator="cuda"))
        assert report.resource.vram_gb == 0, "CPU-only preset must not spend the CUDA VRAM baseline"


class TestModuleCardinality:
    """1 for all-singleton, N for finite lists, None for continuous ranges."""

    def test_all_singleton(self) -> None:
        from autointent.advisor._estimates._search_space import _module_cardinality

        assert _module_cardinality({"module_name": "bert"}) == 1
        assert _module_cardinality({"module_name": "bert", "batch_size": [64], "epochs": [30]}) == 1

    def test_multi_list_multiplies(self) -> None:
        from autointent.advisor._estimates._search_space import _module_cardinality

        # 2 batch x 3 lr candidates = 6 unique configs
        cardinality = _module_cardinality(
            {"module_name": "bert", "batch_size": [32, 64], "learning_rate": [1e-5, 5e-5, 1e-4]}
        )
        assert cardinality == 6

    def test_range_dict_is_unbounded(self) -> None:
        from autointent.advisor._estimates._search_space import _module_cardinality

        # {low, high} → continuous → None (treated as unbounded)
        assert _module_cardinality({"module_name": "knn", "k": {"low": 1, "high": 20}}) is None

    def test_reserved_keys_skipped(self) -> None:
        from autointent.advisor._estimates._search_space import _module_cardinality

        # module_name / target_metric are not search dimensions
        assert (
            _module_cardinality(
                {"module_name": "bert", "target_metric": "scoring_f1", "batch_size": [32, 64]}
            )
            == 2
        )


class TestNoOpHpoFinding:
    """Config-phase warns when n_trials >> unique configs."""

    def test_finding_on_singleton_bert_with_high_n_trials(self) -> None:
        cfg = {
            "search_space": [
                {"node_type": "scoring", "search_space": [
                    {"module_name": "bert",
                     "classification_model_config": [{"model_name": "microsoft/deberta-v3-small"}],
                     "num_train_epochs": [30], "batch_size": [64]},
                ]},
                {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
            ],
            "hpo_config": {"n_trials": 40},
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile())
        no_op = [f for f in report.findings if "unique configurations" in f.message]
        assert len(no_op) == 1, f"expected exactly one no-op warning, got {[f.message for f in no_op]}"
        assert no_op[0].phase == "config"
        assert no_op[0].severity == Severity.TIGHT
        assert "bert" in no_op[0].message

    def test_no_finding_when_search_space_has_range(self) -> None:
        cfg = {
            "search_space": [
                {"node_type": "scoring", "search_space": [
                    {"module_name": "bert",
                     "classification_model_config": [{"model_name": "microsoft/deberta-v3-small"}],
                     "learning_rate": {"low": 1e-5, "high": 1e-4}},
                ]},
                {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
            ],
            "hpo_config": {"n_trials": 40},
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile())
        no_op = [f for f in report.findings if "unique configurations" in f.message]
        assert no_op == [], f"unexpected warning for ranged search space: {[f.message for f in no_op]}"

    def test_no_finding_when_n_trials_matches_cardinality(self) -> None:
        # n_trials=4, cardinality=2x2=4 → not a "no-op" waste
        cfg = {
            "search_space": [
                {"node_type": "scoring", "search_space": [
                    {"module_name": "bert",
                     "classification_model_config": [{"model_name": "microsoft/deberta-v3-small"}],
                     "batch_size": [32, 64], "num_train_epochs": [10, 20]},
                ]},
                {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
            ],
            "hpo_config": {"n_trials": 4},
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile())
        no_op = [f for f in report.findings if "unique configurations" in f.message]
        assert no_op == [], "n_trials matching cardinality should not warn"


class TestModeAwareVramBaseline:
    """CUDA baseline + safety margin are mode-aware — training reserves more
    cuDNN workspace than inference."""

    def test_inference_only_preset_gets_smaller_vram_than_training(self) -> None:
        # Both use e5-large; only the training config triggers the bigger baseline.
        inference_only = {
            "search_space": [
                {"node_type": "scoring", "search_space": [
                    {"module_name": "knn", "k": [5]},
                ]},
                {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
            ],
            "embedder_config": {"model_name": "intfloat/multilingual-e5-large-instruct"},
            "hpo_config": {"n_trials": 5},
        }
        training = {
            "search_space": [
                {"node_type": "scoring", "search_space": [
                    {"module_name": "bert",
                     "classification_model_config": [{"model_name": "microsoft/deberta-v3-small"}],
                     "batch_size": [16], "num_train_epochs": [1]},
                ]},
                {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
            ],
            "hpo_config": {"n_trials": 5},
        }
        stats = DatasetStats.placeholder(n_samples=1000, n_classes=10, avg_tokens=24)

        infer_r = run_preflight(inference_only, stats, _profile(vram_gb=16.0))
        train_r = run_preflight(training, stats, _profile(vram_gb=16.0))

        # The training baseline is 1.0 GB, inference baseline is 0.3 GB — so
        # subtracting the driver max should show at least the 0.7 GB gap.
        infer_max_driver = max((d.get("vram_gb") or 0 for d in infer_r.resource.drivers), default=0)
        train_max_driver = max((d.get("vram_gb") or 0 for d in train_r.resource.drivers), default=0)
        infer_baseline = infer_r.resource.vram_gb - infer_max_driver
        train_baseline = train_r.resource.vram_gb - train_max_driver
        assert infer_baseline < train_baseline, (
            f"inference baseline should be smaller; got infer={infer_baseline:.2f} train={train_baseline:.2f}"
        )
        # Should be roughly the 0.3 vs 1.0 gap (small tolerance for rounding).
        assert train_baseline - infer_baseline > 0.5

    def test_inference_only_still_has_a_cuda_baseline(self) -> None:
        # Even inference-only should be > 0 on CUDA — a non-zero cuDNN + driver
        # context is real. Not zeroing this out would falsely tell users that
        # embedder-only presets need no GPU memory.
        cfg = {
            "search_space": [
                {"node_type": "scoring", "search_space": [
                    {"module_name": "knn", "k": [5]},
                ]},
                {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
            ],
            "embedder_config": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
            "hpo_config": {"n_trials": 5},
        }
        report = run_preflight(cfg, DatasetStats.placeholder(), _profile(vram_gb=16.0))
        assert report.resource.vram_gb > 0
