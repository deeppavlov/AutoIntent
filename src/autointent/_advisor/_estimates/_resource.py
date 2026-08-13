"""Resource-phase orchestration.

Walks the validated search space, asks ``_formulas`` for per-module costs,
aggregates them into a ``ResourceEstimate``, and emits VRAM/RAM/disk/time
findings on the report.

The public entry is ``_resource_phase`` at the bottom; everything above it is
private machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from autointent._advisor import _hub
from autointent._advisor._report import ResourceEstimate, Severity
from autointent.configs._embedder import (
    EmbedderConfig,
    OpenaiEmbeddingConfig,
    SentenceTransformerEmbeddingConfig,
    VllmEmbeddingConfig,
)

from ._formulas import (
    _DEFAULT_SEQ_LEN,
    _LINEAR_CPU_S_PER_SAMPLE_FEATURE,
    _LOGREG_CV_MULTIPLIER,
    _MULTICLASS_THRESHOLD,
    _activations_gb_per_sample,
    _classify_severity,
    _cnn_param_count,
    _embedder_dim,
    _embedder_load_ram_gb,
    _embedding_cache_disk_gb,
    _largest_embedder,
    _max_fitting_batch_size,
    _ram_for_catboost,
    _ram_for_linear,
    _ram_for_module,
    _ram_for_nn,
    _ram_for_sklearn,
    _rnn_param_count,
    _time_for_catboost,
    _time_for_linear,
    _time_for_nn,
    _time_for_transformer,
    _vram_for_nn,
    _vram_for_transformer,
    _weights_vram_for_transformer,
)
from ._search_space import (
    _extract_model_names,
    _max_int,
    _module_cardinality,
    _walk_modules,
    _walk_modules_indexed,
)

if TYPE_CHECKING:
    from autointent._advisor._hardware import HardwareProfile
    from autointent._advisor._hub import ModelMeta
    from autointent._advisor._report import DatasetStats, PreflightReport


# Union variants of EmbedderConfig that carry a model_name attribute.
# HashingVectorizerEmbeddingConfig and the bare BaseEmbedderConfig don't have
# one (sklearn vectorizer / abstract base), so we filter them out below.
_MODEL_BACKED_EMBEDDERS = (
    SentenceTransformerEmbeddingConfig,
    OpenaiEmbeddingConfig,
    VllmEmbeddingConfig,
)


def _embedder_model_name(embedder: EmbedderConfig) -> str | None:
    """Return the embedder's model_name when the config variant carries one."""
    if isinstance(embedder, _MODEL_BACKED_EMBEDDERS):
        return embedder.model_name
    return None


# Maps each fine-tunable transformer module to its training-mode label.
# Modules not listed (or listed as "inference") run the encoder forward-only.
# Note: dnnc keeps the cross-encoder frozen and trains an sklearn LogisticRegressionCV
# head on top of its features (see autointent._wrappers.ranker.Ranker._fit), so the
# encoder's VRAM profile matches inference rather than fine-tuning.
_TRANSFORMER_TRAINING_MODE = {
    "bert": "full-finetune",
    "ptuning": "lora",
    "lora": "lora",
}

# Scorers that consume embeddings (cache key = model + utterances + prompt) but
# don't train the encoder — embedder forward is shared via the persistent cache.
_CACHE_HONORING_MODULES = frozenset(
    {
        "linear",
        "catboost",
        "knn",
        "mlknn",
        "retrieval",
        "description_bi",
        "description_cross",
        "description_llm",
    },
)

# Cache-honoring modules whose per-entry estimate already bundles the embedder
# forward into `time_hours` (vs. classic linear/catboost which don't).
_EMBEDDER_FORWARD_TRANSFORMER_MODULES = frozenset(
    {"knn", "mlknn", "retrieval", "description_bi", "description_cross", "description_llm"},
)


@dataclass
class _ModuleEstimate:
    """Per-module cost contribution + the dict that gets rendered in the report."""

    driver: dict[str, Any]
    vram_gb: float
    ram_gb: float
    time_hours: float
    model_weights_gb: float = 0.0


def _refit_factor(*, refit_after: bool, n_trials: int) -> float:
    """Wall-time multiplier for ``refit_after=True`` (amortized 1/n_trials extra)."""
    return 1 + 1.0 / max(1, n_trials) if refit_after else 1.0


def _split_entries(
    search_space: list[dict[str, Any]],
) -> tuple[list[tuple[int, str, dict[str, Any]]], list[tuple[int, str, dict[str, Any]]]]:
    """Partition search-space entries into (transformer-bearing, classic)."""
    transformer: list[tuple[int, str, dict[str, Any]]] = []
    classic: list[tuple[int, str, dict[str, Any]]] = []
    for node_idx, node_type, entry in _walk_modules_indexed(search_space):
        bucket = classic if entry.get("module_name") in {"linear", "catboost", "sklearn"} else transformer
        bucket.append((node_idx, node_type, entry))
    return transformer, classic


def _estimate_transformer_model(
    *,
    meta: ModelMeta,
    entry: dict[str, Any],
    node_type: str,
    module: str,
    name: str,
    stats: DatasetStats,
    hardware: HardwareProfile,
    n_trials: int,
    refit_after: bool,
) -> _ModuleEstimate:
    """One row of cost for a transformer module + a specific model checkpoint."""
    mode = _TRANSFORMER_TRAINING_MODE.get(module, "inference")
    batch_size = _max_int(entry.get("batch_size"), 32)
    epochs = _max_int(entry.get("num_train_epochs"), 1 if mode == "inference" else 10)
    seq_len = _max_int(entry.get("max_length"), _DEFAULT_SEQ_LEN)

    vram = _vram_for_transformer(meta, mode, batch_size=batch_size, seq_len=seq_len)
    ram = _ram_for_module(meta, stats, mode=mode)

    driver_max_batch: int | None = None
    if hardware.vram_gb > 0:
        driver_max_batch = _max_fitting_batch_size(
            weight_vram_gb=_weights_vram_for_transformer(meta, mode),
            vram_budget_gb=hardware.vram_gb,
            per_sample_gb=_activations_gb_per_sample(meta, seq_len, is_training=mode != "inference"),
        )

    time_h = _time_for_transformer(
        n_trials=n_trials,
        epochs=epochs,
        batch_size=batch_size,
        seq_len=seq_len,
        n_samples=stats.n_samples,
        params_millions=meta.total_params / 1_000_000,
        device_class=hardware.device_class,
    )
    if mode != "inference":
        time_h *= _refit_factor(refit_after=refit_after, n_trials=n_trials)

    return _ModuleEstimate(
        driver={
            "node_type": node_type,
            "module": module,
            "model": name,
            "mode": mode,
            "vram_gb": round(vram, 2),
            "ram_gb": round(ram, 2),
            "time_hours": round(time_h, 2),
            "batch_size": batch_size,
            "max_batch_size": driver_max_batch,
            "confidence": meta.confidence,
        },
        vram_gb=vram,
        ram_gb=ram,
        time_hours=time_h,
        model_weights_gb=meta.weights_gb,
    )


def _estimate_classic_entry(
    *,
    entry: dict[str, Any],
    node_type: str,
    embedder_meta: ModelMeta | None,
    embedder_dim: int,
    stats: DatasetStats,
    hardware: HardwareProfile,
    n_trials: int,
    refit_after: bool,
) -> _ModuleEstimate | None:
    """Cost row for a linear or catboost scorer (returns ``None`` for any other module)."""
    module = entry.get("module_name", "?")
    refit = _refit_factor(refit_after=refit_after, n_trials=n_trials)
    # Both multinomial (multiclass) and one-vs-rest (multilabel) LR scale linearly in n_classes.
    class_multiplier = max(1, stats.n_classes)

    if module == "linear":
        cv_multiplier = 1 if stats.multilabel else _LOGREG_CV_MULTIPLIER
        ram = _ram_for_linear(stats=stats, embedder_dim=embedder_dim)
        time_h = (
            _time_for_linear(
                n_trials=n_trials,
                n_samples=stats.n_samples,
                embedder_dim=embedder_dim,
                max_iter=_max_int(entry.get("max_iter"), 100),
                cv_multiplier=cv_multiplier,
                class_multiplier=class_multiplier,
            )
            * refit
        )
        vram = 0.0
        mode = "linear-cv" if cv_multiplier > 1 else "linear"
    elif module == "catboost":
        on_gpu = entry.get("task_type") == "GPU" and hardware.accelerator == "cuda"
        # CatBoost MultiClass loss grows per-class trees only above binary; binary uses
        # Logloss with one tree per iteration.
        cb_class_mult = class_multiplier if stats.n_classes > _MULTICLASS_THRESHOLD or stats.multilabel else 1
        iterations = _max_int(entry.get("iterations"), 1000)
        depth = _max_int(entry.get("depth"), 6)
        ram_total = _ram_for_catboost(stats=stats, n_features=embedder_dim, iterations=iterations, depth=depth)
        time_h = (
            _time_for_catboost(
                n_trials=n_trials,
                n_samples=stats.n_samples,
                n_features=embedder_dim,
                iterations=iterations,
                depth=depth,
                class_multiplier=cb_class_mult,
                on_gpu=on_gpu,
            )
            * refit
        )
        vram, ram = (ram_total, 0.0) if on_gpu else (0.0, ram_total)
        mode = "catboost-gpu" if on_gpu else "catboost"
    elif module == "sklearn":
        # RandomForestClassifier is the most common target; joblib spawns
        # ``n_jobs`` worker processes each replicating the feature matrix +
        # trees. Predicting 0 here (previous "not-estimated" behaviour) hid
        # classic-heavy's real 1-2 GB sklearn contribution.
        n_estimators = _max_int(entry.get("n_estimators"), 100)
        max_depth = _max_int(entry.get("max_depth"), 0)
        sk_n_jobs = _max_int(entry.get("n_jobs"), 1)
        ram = _ram_for_sklearn(
            stats=stats,
            embedder_dim=embedder_dim,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=sk_n_jobs,
        )
        # Time: rough O(n_estimators × n_samples × sqrt(features) × log2 n)
        # per fit, divided by n_jobs. Absorbed into the linear coefficient
        # since real numbers vary wildly by criterion / max_features.
        time_h = (
            n_trials
            * _LINEAR_CPU_S_PER_SAMPLE_FEATURE
            * stats.n_samples
            * embedder_dim
            * n_estimators
            / max(1, sk_n_jobs)
            / 3600.0
        ) * refit
        vram = 0.0
        mode = f"sklearn-n_jobs={sk_n_jobs}"
    else:
        return None

    return _ModuleEstimate(
        driver={
            "node_type": node_type,
            "module": module,
            "model": embedder_meta.name if embedder_meta else "(no embedder)",
            "mode": mode,
            "vram_gb": round(vram, 2),
            "ram_gb": round(ram, 2),
            "time_hours": round(time_h, 2),
            "batch_size": None,
            "max_batch_size": None,
            "confidence": embedder_meta.confidence if embedder_meta else "heuristic",
        },
        vram_gb=vram,
        ram_gb=ram,
        time_hours=time_h,
    )


def _estimate_nn_entry(
    *,
    entry: dict[str, Any],
    node_type: str,
    stats: DatasetStats,
    hardware: HardwareProfile,
    n_trials: int,
    refit_after: bool,
) -> _ModuleEstimate | None:
    """Cost row for a cnn / rnn scorer (returns ``None`` for anything else).

    These are small torch models trained from scratch on token ids. Previously
    the advisor emitted a ``not-estimated`` placeholder for them, which read
    as "free/safe" — nn-heavy on banking77 predicted 0h/0GB but actually used
    0.32 h + 2.3 GB RAM + 0.7 GB VRAM. This restores a real estimate using
    small-model parameter counts + the transformer FLOPs formula for time.
    """
    module = entry.get("module_name", "?")
    n_classes = max(1, stats.n_classes)

    if module == "cnn":
        embed_dim = _max_int(entry.get("embed_dim"), 128)
        num_filters = _max_int(entry.get("num_filters"), 100)
        kernel_sizes = entry.get("kernel_sizes")
        # Kernel sizes are a list of lists in the search space
        # (e.g. [[3, 4, 5]]); count entries in the largest variant.
        n_kernels = 3
        if isinstance(kernel_sizes, list):
            for candidate in kernel_sizes:
                if isinstance(candidate, list):
                    n_kernels = max(n_kernels, len(candidate))
                elif isinstance(candidate, int):
                    n_kernels = max(n_kernels, 1)
        hidden_dim = num_filters
        params = _cnn_param_count(
            embed_dim=embed_dim, num_filters=num_filters, n_kernels=n_kernels, n_classes=n_classes
        )
    elif module == "rnn":
        embed_dim = _max_int(entry.get("embed_dim"), 128)
        hidden_dim = _max_int(entry.get("hidden_dim"), 512)
        params = _rnn_param_count(embed_dim=embed_dim, hidden_dim=hidden_dim, n_classes=n_classes)
    else:
        return None

    batch_size = _max_int(entry.get("batch_size"), 64)
    epochs = _max_int(entry.get("num_train_epochs"), 60)

    vram = _vram_for_nn(params=params, batch_size=batch_size, hidden_dim=hidden_dim)
    ram = _ram_for_nn(params=params, stats=stats)
    time_h = (
        _time_for_nn(
            n_trials=n_trials,
            epochs=epochs,
            batch_size=batch_size,
            n_samples=stats.n_samples,
            params_millions=params / 1_000_000,
            device_class=hardware.device_class,
        )
        * _refit_factor(refit_after=refit_after, n_trials=n_trials)
    )

    return _ModuleEstimate(
        driver={
            "node_type": node_type,
            "module": module,
            "model": f"{module}-from-scratch",
            "mode": "small-torch-train",
            "vram_gb": round(vram, 2),
            "ram_gb": round(ram, 2),
            "time_hours": round(time_h, 2),
            "batch_size": batch_size,
            "max_batch_size": None,
            "confidence": "heuristic",
        },
        vram_gb=vram,
        ram_gb=ram,
        time_hours=time_h,
    )


def _apply_embedding_cache(
    module_estimates: list[_ModuleEstimate],
    seen_models: dict[str, ModelMeta],
    *,
    stats: DatasetStats,
    hardware: HardwareProfile,
    cache_probe: Callable[[str], bool] | None = None,
) -> set[str]:
    """Adjust ``module_estimates`` in-place for autointent's persistent embedding cache.

    Per unique embedder, the first cache-honoring entry pays the forward; later
    transformer entries get ``time_hours`` zeroed (cache hit), and classic
    entries (linear/catboost) get a synthetic forward added since their
    per-entry estimate doesn't include one.

    ``cache_probe`` (optional): callable that takes an embedder model_name and
    returns True if the embedding is already cached on disk. When it returns
    True, the model is treated as pre-paid — forward is zero and disk cache
    delta is zero. Default (None) preserves the pessimistic cold assumption
    the advisor shipped with — every embedder pays once.

    Returns the set of unique embedder model names whose forward was charged
    (i.e. contributed to ``disk_embedding_cache_gb`` in the disk aggregation).
    """
    paid: set[str] = set()
    # Models the probe reports as already-warm — pre-populate ``paid`` so the
    # first-seen module also hits the cache-hit branch instead of paying the
    # forward, and skip them in the disk-cache aggregation (already on disk).
    warm_models: set[str] = set()
    if cache_probe is not None:
        for name in seen_models:
            if cache_probe(name):
                warm_models.add(name)
    for me in module_estimates:
        module = me.driver["module"]
        if module not in _CACHE_HONORING_MODULES:
            continue
        model = me.driver["model"]
        if model not in seen_models:  # synthetic / "(no embedder)" rows
            continue
        if model in warm_models:
            if module in _EMBEDDER_FORWARD_TRANSFORMER_MODULES:
                me.time_hours = 0.0
                me.driver["time_hours"] = 0.0
                me.driver["mode"] = f"{me.driver['mode']}+warm"
            continue
        if model in paid:
            if module in _EMBEDDER_FORWARD_TRANSFORMER_MODULES:
                me.time_hours = 0.0
                me.driver["time_hours"] = 0.0
                me.driver["mode"] = f"{me.driver['mode']}+cached"
        else:
            paid.add(model)
            if module in {"linear", "catboost"}:
                embedder_meta = seen_models.get(model)
                forward_h = _time_for_transformer(
                    n_trials=1,
                    epochs=1,
                    batch_size=32,
                    seq_len=128,
                    n_samples=stats.n_samples,
                    params_millions=(embedder_meta.total_params / 1_000_000) if embedder_meta else 100.0,
                    device_class=hardware.device_class,
                )
                me.time_hours += forward_h
                me.driver["time_hours"] = round(me.time_hours, 2)
                me.driver["mode"] = f"{me.driver['mode']}+embed"
    return paid


def _aggregate_disk(
    estimate: ResourceEstimate,
    seen_models: dict[str, ModelMeta],
    node_max_weights: dict[int, float],
    *,
    dump_modules: bool,
    n_trials: int,
    cached_embedders: set[str] | None = None,
    stats: DatasetStats | None = None,
) -> None:
    """Fold per-model download/cached/embedding-cache sizes into ``estimate``."""
    for meta in seen_models.values():
        if meta.cached_locally:
            estimate.disk_cached_gb += meta.disk_gb
        else:
            estimate.disk_download_gb += meta.disk_gb
    if dump_modules:
        # Each trial selects one variant per node, so per-trial dumped weights
        # are bounded by the heaviest module in each node, summed across nodes.
        estimate.disk_dump_gb = sum(node_max_weights.values()) * n_trials

    if cached_embedders and stats is not None:
        for name in cached_embedders:
            meta = seen_models.get(name)
            if meta is None:
                continue
            estimate.disk_embedding_cache_gb += _embedding_cache_disk_gb(
                n_samples=stats.n_samples,
                hidden_size=_embedder_dim(meta),
            )


def _emit_resource_findings(
    report: PreflightReport,
    estimate: ResourceEstimate,
    hardware: HardwareProfile,
    *,
    n_jobs: int,
) -> None:
    """Translate aggregated estimates into VRAM/RAM/disk/time findings on the report."""
    parallel_gpu = n_jobs > 1 and hardware.accelerator in {"cuda", "mps"}
    effective_vram = estimate.vram_gb * n_jobs if parallel_gpu else estimate.vram_gb
    # MPS shares one unified pool: parallel workers each allocate weights+activations
    # in RAM, so peak RAM also scales with n_jobs on Apple Silicon.
    effective_ram = estimate.ram_gb * n_jobs if n_jobs > 1 and hardware.accelerator == "mps" else estimate.ram_gb

    if hardware.accelerator == "cpu" and effective_vram > 0:
        report.add(
            "resource",
            Severity.TIGHT,
            f"No GPU detected; transformer modules will be very slow (worst case ~{estimate.time_hours:.1f} h).",
            metric="vram",
        )
    else:
        msg = f"VRAM ~{effective_vram:.1f} GB"
        if n_jobs > 1:
            msg += f" (= per-trial {estimate.vram_gb:.1f} GB x {n_jobs} parallel trials)"
        msg += f" vs available {hardware.vram_gb:.1f} GB"
        report.add("resource", _classify_severity(effective_vram, hardware.vram_gb), msg, metric="vram")

    report.add(
        "resource",
        _classify_severity(effective_ram, hardware.ram_gb),
        f"RAM ~{effective_ram:.1f} GB vs available {hardware.ram_gb:.1f} GB",
        metric="ram",
    )

    disk_total = estimate.disk_download_gb + estimate.disk_dump_gb + estimate.disk_embedding_cache_gb
    disk_msg = f"Disk ~{estimate.disk_download_gb:.1f} GB to download"
    if estimate.disk_cached_gb > 0:
        disk_msg += f", {estimate.disk_cached_gb:.1f} GB already cached"
    if estimate.disk_dump_gb > 0:
        disk_msg += f", +{estimate.disk_dump_gb:.1f} GB during training (dump_modules=True)"
    if estimate.disk_embedding_cache_gb > 0:
        disk_msg += f", +{estimate.disk_embedding_cache_gb:.2f} GB embedding cache"
    disk_msg += f" vs {hardware.free_disk_gb:.0f} GB free"
    report.add("resource", _classify_severity(disk_total, hardware.free_disk_gb), disk_msg, metric="disk")

    if estimate.time_hours > 0:
        report.add(
            "resource",
            Severity.AMPLE,
            f"Time ~{estimate.time_hours:.1f} h (worst case, no HPO pruning)",
            metric="time",
        )


# Process-level memory floors. Every autointent fit imports torch +
# transformers + datasets + optuna, which reserve resident memory the moment
# they load. Measured against calibration_runs2 (2026-08-07 banking77 sweep):
# every preset used 2-10 GB RAM but the per-module estimates alone predicted
# 0.4-2 GB → advisor was systematically 4-13x LOW on RAM. A ~1.5 GB baseline
# lifts the estimate into range without over-inflating heavy presets.
_PROCESS_BASELINE_RAM_GB = 1.5
# CUDA driver context + cuDNN/cuBLAS workspace pools + caching allocator
# fragmentation. A real training process on A100 reserves ~1 GB the moment
# torch initializes CUDA + the first tensor lands, regardless of model size.
# Earlier 0.5 GB baseline left transformers-light on banking77 at 8.12 GB
# predicted vs 8.75 GB measured (unsafe under-prediction for OOM avoidance);
# 1.0 GB closes the gap with room to spare. Only added when we already
# predict some VRAM usage so CPU-only presets aren't spuriously flagged as
# GPU users.
_CUDA_BASELINE_VRAM_GB = 1.0

_UNKNOWN_SCORER_MODULES: frozenset[str] = frozenset()
"""Scorer modules the advisor has no cost estimator for — kept as an empty
extension point. cnn / rnn moved to :func:`_estimate_nn_entry`; sklearn moved
to the classic branch of :func:`_estimate_classic_entry`. New unknown-cost
scorers should still register here so they emit a not-estimated placeholder
row instead of silently reporting zero."""

_NN_SCORER_MODULES = frozenset({"cnn", "rnn"})
"""Small torch scorers routed through :func:`_estimate_nn_entry`."""


_EMBEDDER_CONSUMING_MODULES = frozenset(
    {"linear", "catboost", "sklearn", "knn", "mlknn", "retrieval",
     "description_bi", "description_cross", "description_llm"},
)


def _uses_embedder(search_space: list[dict[str, Any]]) -> bool:
    """True when any module in the search space consumes an embedder — signals
    that the aggregate RAM should include the embedder-load penalty on top
    of the per-driver max."""
    for _, entry in _walk_modules(search_space):
        if entry.get("module_name") in _EMBEDDER_CONSUMING_MODULES:
            return True
    return False

# Modules that consume the top-level ``cross_encoder_config.model_name`` as
# their scoring model (see zero-shot-encoders preset: description_cross pulls
# BAAI/bge-reranker-v2-m3 from that config, not from its per-entry dict).
_CROSS_ENCODER_CONSUMERS = frozenset({"description_cross", "dnnc", "retrieval"})

# Modules that fall back to the top-level ``transformer_config.model_name``
# when no per-entry ``classification_model_config`` is given.
_TRANSFORMER_CONFIG_CONSUMERS = frozenset({"bert"})


def _not_estimated_row(*, node_type: str, module: str) -> _ModuleEstimate:
    """Placeholder row for a module the advisor has no cost formula for.

    Renders as ``mode="not-estimated"`` in the report so the module isn't
    silently absent (would read as "free/safe") — a call to action for whoever
    reads the JSON that the actual cost is unknown, not zero.
    """
    return _ModuleEstimate(
        driver={
            "node_type": node_type,
            "module": module,
            "model": "(not estimated)",
            "mode": "not-estimated",
            "vram_gb": 0.0,
            "ram_gb": 0.0,
            "time_hours": 0.0,
            "batch_size": None,
            "max_batch_size": None,
            "confidence": "unknown",
            "note": "advisor has no cost estimator for this module; treat as unknown, not zero",
        },
        vram_gb=0.0,
        ram_gb=0.0,
        time_hours=0.0,
    )


def _resource_phase(
    *,
    embedder_config: EmbedderConfig,
    search_space: list[dict[str, Any]],
    n_trials: int,
    n_jobs: int,
    dump_modules: bool,
    stats: DatasetStats,
    hardware: HardwareProfile,
    report: PreflightReport,
    refit_after: bool = False,
    cross_encoder_model_name: str | None = None,
    transformer_model_name: str | None = None,
    cache_probe: Callable[[str], bool] | None = None,
) -> None:
    """Walk the validated search space, fold per-module costs into the report.

    Two passes: transformer-bearing modules first (collects ``seen_models`` so
    the largest model can drive ``embedder_dim`` for the classic pass), then
    linear / catboost. Disk, VRAM/RAM peak, time sum, and final findings are
    folded onto the report.

    ``cross_encoder_model_name`` and ``transformer_model_name`` come from the
    pipeline's top-level configs. They're used as the fallback model for
    modules that don't declare a per-entry ``classification_model_config`` but
    still consume one at runtime (``description_cross`` / ``dnnc`` /
    ``retrieval`` pull from ``cross_encoder_config``; ``bert`` falls back to
    ``transformer_config``). Seeding them here fixes the disk-download
    under-count called out in the follow-up review (missing 6.4 GB reranker in
    ``zero-shot-encoders``).
    """
    seen_models: dict[str, ModelMeta] = {}
    global_embedder = _embedder_model_name(embedder_config)
    if global_embedder:
        seen_models[global_embedder] = _hub.resolve_model(global_embedder)

    transformer_entries, classic_entries = _split_entries(search_space)

    # Per-node module-variant count. HPO distributes ``n_trials`` across the
    # module_name candidates at each node roughly evenly (TPE's sampler bias
    # aside), so a module_name that shares its node with 4 others sees on
    # average ``n_trials / 5`` trials — not ``n_trials``. Previously every
    # per-module estimate used the full ``n_trials``, which inflated
    # classic-heavy's catboost row to 120 h vs measured 12 h across the whole
    # node. Divide once here and pass the effective share down to every
    # per-module estimator (transformer + classic + nn).
    variants_per_node: dict[int, int] = {}
    for node_idx, _node_type, _entry in _walk_modules_indexed(search_space):
        variants_per_node[node_idx] = variants_per_node.get(node_idx, 0) + 1

    def _effective_trials(node_idx: int, entry: dict[str, Any] | None = None) -> int:  # noqa: ARG001
        """Trials this specific module should be charged for.

        Divides ``n_trials`` evenly across the module_name candidates at the
        node. ``entry`` is accepted for API stability and future extensions
        (e.g. a per-module cardinality cap) — we tried capping by
        :func:`_module_cardinality` earlier but it produced wrong estimates
        for description-scorer presets where Optuna's TPE runs every
        declared trial regardless of parameter-space size (no automatic
        dedup). The advisor charges for the declared work; if a real run
        crashes or dedupes, that's an artefact of the runtime, not something
        the advisor should try to predict.
        """
        divisor = max(1, variants_per_node.get(node_idx, 1))
        return max(1, n_trials // divisor)

    # First pass: transformer modules (also populates seen_models for the classic pass).
    module_estimates: list[_ModuleEstimate] = []
    node_max_weights: dict[int, float] = {}
    for node_idx, node_type, entry in transformer_entries:
        module = entry.get("module_name", "?")
        model_names = _extract_model_names(entry)
        if not model_names:
            if module in {"knn", "mlknn"} and global_embedder:
                model_names = [global_embedder]
            elif module in _CROSS_ENCODER_CONSUMERS and cross_encoder_model_name:
                model_names = [cross_encoder_model_name]
            elif module in _TRANSFORMER_CONFIG_CONSUMERS and transformer_model_name:
                model_names = [transformer_model_name]
            elif module in _NN_SCORER_MODULES:
                # cnn / rnn — small torch models trained from scratch, no hub
                # model to resolve. Route to the small-model heuristic.
                nn_estimate = _estimate_nn_entry(
                    entry=entry,
                    node_type=node_type,
                    stats=stats,
                    hardware=hardware,
                    n_trials=_effective_trials(node_idx, entry),
                    refit_after=refit_after,
                )
                if nn_estimate is not None:
                    module_estimates.append(nn_estimate)
                continue
            elif module in _UNKNOWN_SCORER_MODULES:
                # Placeholder so the row is visible instead of silently zeroed.
                module_estimates.append(_not_estimated_row(node_type=node_type, module=module))
                continue
        for name in model_names:
            meta = seen_models.setdefault(name, _hub.resolve_model(name))
            me = _estimate_transformer_model(
                meta=meta,
                entry=entry,
                node_type=node_type,
                module=module,
                name=name,
                stats=stats,
                hardware=hardware,
                n_trials=_effective_trials(node_idx, entry),
                refit_after=refit_after,
            )
            module_estimates.append(me)
            # Track heaviest weight per node so dump_modules is bounded by one
            # selected variant per node x n_trials, not the sum of all candidates.
            node_max_weights[node_idx] = max(node_max_weights.get(node_idx, 0.0), me.model_weights_gb)

    # Second pass: linear / catboost — cost depends on embedder_dim, not a checkpoint.
    embedder_meta = _largest_embedder(seen_models)
    embedder_dim_val = _embedder_dim(embedder_meta)
    for node_idx, node_type, entry in classic_entries:
        classic_estimate = _estimate_classic_entry(
            entry=entry,
            node_type=node_type,
            embedder_meta=embedder_meta,
            embedder_dim=embedder_dim_val,
            stats=stats,
            hardware=hardware,
            n_trials=_effective_trials(node_idx, entry),
            refit_after=refit_after,
        )
        if classic_estimate is not None:
            module_estimates.append(classic_estimate)

    # Cache-aware time/disk: must run before the fold below.
    cached_embedders = _apply_embedding_cache(
        module_estimates, seen_models, stats=stats, hardware=hardware, cache_probe=cache_probe,
    )

    estimate = ResourceEstimate(parallel_factor=n_jobs)
    for me in module_estimates:
        estimate.vram_gb = max(estimate.vram_gb, me.vram_gb)
        estimate.ram_gb = max(estimate.ram_gb, me.ram_gb)
        estimate.time_hours += me.time_hours
        estimate.drivers.append(me.driver)

    # Process-level baselines. Every fit — even a trivial one — imports
    # torch / transformers / datasets, which on their own take ~1.5 GB of RSS
    # before any model weights load. Real runs on banking77 measure 2-10 GB
    # RAM across every preset while the per-module estimates alone predicted
    # 0.4-2 GB (systematically 4-13x low, see calibration_runs2 2026-08-07).
    # Adding a floor here (rather than per-module) means the estimate stays
    # accurate when multiple modules coexist — the floor is paid once, not N times.
    estimate.ram_gb = max(estimate.ram_gb, 0.0) + _PROCESS_BASELINE_RAM_GB
    # Embedder-load penalty for classic presets: when at least one classic
    # scorer (linear / catboost / sklearn / knn / mlknn) sits on top of an
    # embedder, the process holds the embedder weights + tokenizer + HF
    # buffers *in addition to* whatever the per-driver ``max`` reported.
    # classic-heavy on banking77 predicted 2.86 GB RAM against a measured
    # 10.22 GB (3.6x under) because the max-of-drivers hides the fact that
    # multiple scorers coexist in RAM. Only added when the search space
    # actually consumes an embedder.
    if _uses_embedder(search_space) and embedder_meta is not None:
        estimate.ram_gb += _embedder_load_ram_gb(embedder_meta)
    # Same story on the CUDA side: torch's caching allocator, cuBLAS/cuDNN
    # workspaces, and driver context together reserve ~0.5 GB the moment the
    # first tensor lands on the device — regardless of model size. Only apply
    # when we actually predict some GPU usage AND running on CUDA hardware,
    # so CPU-only presets stay honest.
    if estimate.vram_gb > 0 and hardware.accelerator == "cuda":
        estimate.vram_gb += _CUDA_BASELINE_VRAM_GB

    _aggregate_disk(
        estimate,
        seen_models,
        node_max_weights,
        dump_modules=dump_modules,
        n_trials=n_trials,
        cached_embedders=cached_embedders,
        stats=stats,
    )

    # Flip low_confidence if any model fell back to the heuristic path (Hub
    # unreachable, repo missing safetensors metadata, local-path checkpoint).
    # Emit as a TIGHT finding (not just a note) so it shows up in the main
    # rendered findings block — buried notes previously let ~2× under-prediction
    # of large-model shapes slip past the reviewer.
    heuristic_models = [m.name for m in seen_models.values() if m.confidence == "heuristic"]
    if heuristic_models:
        report.low_confidence = True
        sample = ", ".join(heuristic_models[:3]) + ("..." if len(heuristic_models) > 3 else "")  # noqa: PLR2004
        report.add(
            "resource",
            Severity.TIGHT,
            f"LOW CONFIDENCE - Hub metadata unavailable for {len(heuristic_models)} model(s); "
            f"cost estimates use conservative large-model defaults (may over-predict small models): {sample}",
        )

    report.resource = estimate
    _emit_resource_findings(report, estimate, hardware, n_jobs=n_jobs)
