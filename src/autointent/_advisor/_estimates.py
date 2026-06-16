"""Resource-phase estimation: walk the search space and aggregate cost.

Implements an honest worst-case for the modules the proposal lists as
in-scope. Formulas are intentionally coarse — the advisor's contract is
"heuristic upper bound, not measurement". Time and VRAM are the noisiest;
treat them as ballparks, not budgets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from autointent.configs._optimization import HPOConfig

from ._hub import hub_reachable, resolve_model
from ._report import PreflightReport, ResourceEstimate, Severity

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._hardware import HardwareProfile
    from ._hub import ModelMeta
    from ._report import DatasetStats

_MULTICLASS_THRESHOLD = 2

# Fallback architecture shape (BERT-base) used only when the model's actual
# config.json couldn't be fetched from HF Hub — see _hub._shape_from_config.
_DEFAULT_HIDDEN = 768
_DEFAULT_LAYERS = 12

logger = logging.getLogger(__name__)


class _AdvisorConfig(BaseModel):
    """Validated view of the advisor's input config.

    Wraps the four top-level keys the phase helpers read. Unknown top-level
    keys are ignored (preset YAMLs carry extra metadata the advisor doesn't model).
    """

    model_config = ConfigDict(extra="ignore")

    hpo_config: HPOConfig = Field(default_factory=HPOConfig)
    search_space: list[dict[str, Any]] = Field(default_factory=list)
    refit_after: bool = False
    dump_modules: bool = False
    embedder_config: dict[str, Any] | None = None


def _validated_config(config: dict[str, Any]) -> _AdvisorConfig:
    """Validate ``config`` against ``_AdvisorConfig``; fall back to defaults on any error.

    The advisor is best-effort: a malformed user config should still produce a
    report (with placeholder costs) rather than crashing.
    """
    try:
        return _AdvisorConfig.model_validate(config)
    except ValidationError as e:
        logger.warning("Advisor config failed validation; falling back to defaults: %s", e)
        return _AdvisorConfig()


# Severity thresholds as a fraction of available budget: at or above _TIGHT
# downgrades to Severity.TIGHT; at or above _OVER downgrades to Severity.OVER.
_TIGHT_RATIO = 0.7
_OVER_RATIO = 1.0

# rough per-step seconds, keyed on device class. Scaled by params_millions / 100.
_PER_STEP_BASELINE_S = {
    "cpu": 0.5,
    "low-gpu": 0.04,
    "mid-gpu": 0.02,
    "high-gpu": 0.01,
    "apple-silicon": 0.08,
}

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

# Fallback max_length when the search-space entry doesn't pin it. Used both as
# the default in _vram_for_transformer and in the entry-walk seq_len resolution.
_DEFAULT_SEQ_LEN = 128

# Coefficients for the linear / catboost time formulas (proposal §"Algorithm").
_LINEAR_CPU_S_PER_SAMPLE_FEATURE_ITER = 1e-8
_CATBOOST_CPU_S_PER_SAMPLE_FEATURE_ITER = 1e-9
_CATBOOST_GPU_SPEEDUP = 10.0
# LogisticRegressionCV defaults: Cs=10, cv=3 → 31 inner fits + 1 final refit.
_LOGREG_CV_MULTIPLIER = 31
_CATBOOST_DEFAULT_BINS = 254
# Bytes per histogram bucket / tree node — order-of-magnitude constants.
_CATBOOST_BYTES_PER_TREE_NODE = 32


def _extract_model_names(module_entry: dict[str, Any]) -> list[str]:
    """Pull model name(s) from a search-space module entry."""
    candidates: list[str] = []
    cfg = module_entry.get("classification_model_config")
    if isinstance(cfg, list):
        candidates.extend(c["model_name"] for c in cfg if isinstance(c, dict) and c.get("model_name"))
    elif isinstance(cfg, dict) and cfg.get("model_name"):
        candidates.append(cfg["model_name"])
    embedder_cfg = module_entry.get("embedder_config")
    if isinstance(embedder_cfg, list):
        candidates.extend(c["model_name"] for c in embedder_cfg if isinstance(c, dict) and c.get("model_name"))
    elif isinstance(embedder_cfg, dict) and embedder_cfg.get("model_name"):
        candidates.append(embedder_cfg["model_name"])
    return candidates


def _max_int(value: Any, default: int) -> int:  # noqa: ANN401
    if value is None:
        return default
    if isinstance(value, list) and value:
        return max(int(x) for x in value)
    if isinstance(value, dict):
        return int(value.get("high", default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _walk_modules_indexed(
    search_space: list[dict[str, Any]],
) -> Iterable[tuple[int, str, dict[str, Any]]]:
    """Yield (node_index, node_type, module_entry) — index lets us bound per-node max cost."""
    for node_idx, node in enumerate(search_space or []):
        node_type = node.get("node_type", "?")
        for entry in node.get("search_space", []) or []:
            yield node_idx, node_type, entry


def _walk_modules(search_space: list[dict[str, Any]]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield (node_type, module_entry) pairs — index-agnostic view over `_walk_modules_indexed`."""
    for _, node_type, entry in _walk_modules_indexed(search_space):
        yield node_type, entry


def _weights_vram_for_transformer(meta: ModelMeta, mode: str) -> float:
    """Weight-side VRAM in GB — weights + grads + Adam optimizer state. Excludes activations.

    Modes:
      * ``inference``: forward only — weights + ~30% intermediate-tensor overhead.
      * ``lora``: frozen base + small trainable adapters + their grads/optimizer (~0.5 GB).
      * ``full-finetune`` (default): weights + grads + Adam (m, v) = 4x weights.
    """
    weights_gb = meta.weights_gb
    if mode == "inference":
        return weights_gb * 1.3
    if mode == "lora":
        return weights_gb * 1.3 + 0.5
    return weights_gb * 4.0


def _vram_for_transformer(
    meta: ModelMeta,
    mode: str,
    mixed_precision: bool,
    *,
    batch_size: int = 0,
    seq_len: int = _DEFAULT_SEQ_LEN,
) -> float:
    """Total VRAM in GB: weights + grads + optimizer state + activations x batch.

    Activation accounting differs by mode — training keeps per-layer outputs for
    backward; inference only needs one or two layers in flight.
    """
    base = _weights_vram_for_transformer(meta, mode)
    if batch_size <= 0:
        return base
    per_sample = _activations_gb_per_sample(
        meta, seq_len, mixed_precision=mixed_precision, is_training=mode != "inference"
    )
    return base + per_sample * batch_size


def _ram_for_module(meta: ModelMeta, stats: DatasetStats) -> float:
    """RAM in GB. Loose upper bound."""
    return meta.weights_gb + (stats.n_samples * stats.avg_tokens * 4) / (1024**3)


def _floor_to_power_of_two(n: int) -> int:
    """Largest power of two ≤ n; returns 0 when n < 1."""
    if n < 1:
        return 0
    power = 1
    while power * 2 <= n:
        power *= 2
    return power


def _n_layers(meta: ModelMeta | None) -> int:
    """Layer count from the model's ``config.json``; falls back to BERT-base when absent."""
    if meta is not None and meta.n_layers is not None:
        return meta.n_layers
    return _DEFAULT_LAYERS


def _activations_gb_per_sample(
    meta: ModelMeta | None,
    seq_len: int,
    *,
    mixed_precision: bool,
    is_training: bool,
) -> float:
    """Heuristic activation memory per sample.

    Training: ``seq_len x hidden x layers x const`` — per-layer outputs are kept
    for backward.
    Inference: ``seq_len x hidden x const`` — only one or two layers' outputs in
    flight at once.
    Mixed precision halves activation bytes.
    """
    hidden = _embedder_dim(meta)
    # Training keeps every layer's outputs for backward -> scales x n_layers.
    # The 16-byte/token/layer coefficient bundles fp32 activation + ~4x backward overhead.
    # Inference only holds ~1-2 layers' outputs in flight at once.
    bytes_per_sample = seq_len * hidden * _n_layers(meta) * 16 if is_training else seq_len * hidden * 8
    if mixed_precision:
        bytes_per_sample //= 2
    return bytes_per_sample / (1024**3)


def _max_fitting_batch_size(
    *,
    weight_vram_gb: float,
    vram_budget_gb: float,
    per_sample_gb: float,
) -> int:
    """Largest batch that keeps total VRAM under the AMPLE/TIGHT threshold.

    Returns 0 when even the weights blow the budget. Result is rounded down to
    the nearest power of two.
    """
    if per_sample_gb <= 0:
        return 0
    target_vram = vram_budget_gb * _TIGHT_RATIO
    available_for_activations = target_vram - weight_vram_gb
    if available_for_activations <= 0:
        return 0
    return _floor_to_power_of_two(int(available_for_activations / per_sample_gb))


def _embedder_dim(meta: ModelMeta | None) -> int:
    """Hidden size from the model's ``config.json``; falls back to BERT-base when absent."""
    if meta is not None and meta.hidden_size is not None:
        return meta.hidden_size
    return _DEFAULT_HIDDEN


def _largest_embedder(seen_models: dict[str, ModelMeta]) -> ModelMeta | None:
    if not seen_models:
        return None
    return max(seen_models.values(), key=lambda m: m.params_millions)


def _ram_for_linear(*, stats: DatasetStats, embedder_dim: int) -> float:
    """Float64 design matrix dominates; coefficients and L-BFGS history are small."""
    data_bytes = 8.0 * stats.n_samples * embedder_dim
    coef_bytes = 8.0 * max(1, stats.n_classes) * embedder_dim
    lbfgs_bytes = 10.0 * 8.0 * embedder_dim
    return (data_bytes + coef_bytes + lbfgs_bytes) / (1024**3)


def _time_for_linear(
    *,
    n_trials: int,
    n_samples: int,
    embedder_dim: int,
    max_iter: int,
    cv_multiplier: int,
    class_multiplier: int,
) -> float:
    seconds = (
        n_trials
        * _LINEAR_CPU_S_PER_SAMPLE_FEATURE_ITER
        * n_samples
        * embedder_dim
        * max_iter
        * cv_multiplier
        * class_multiplier
    )
    return seconds / 3600.0


def _ram_for_catboost(*, stats: DatasetStats, n_features: int, iterations: int, depth: int) -> float:
    data_bytes = 4.0 * stats.n_samples * n_features
    histograms_bytes = 4.0 * n_features * _CATBOOST_DEFAULT_BINS
    trees_bytes = iterations * (2**depth) * _CATBOOST_BYTES_PER_TREE_NODE
    return float((data_bytes + histograms_bytes + trees_bytes) / (1024**3))


def _time_for_catboost(
    *,
    n_trials: int,
    n_samples: int,
    n_features: int,
    iterations: int,
    depth: int,
    class_multiplier: int,
    on_gpu: bool,
) -> float:
    coeff = _CATBOOST_CPU_S_PER_SAMPLE_FEATURE_ITER
    if on_gpu:
        coeff /= _CATBOOST_GPU_SPEEDUP
    seconds = n_trials * iterations * coeff * n_samples * n_features * depth * class_multiplier
    return seconds / 3600.0


def _time_for_transformer(
    *,
    meta: ModelMeta,
    n_trials: int,
    epochs: int,
    batch_size: int,
    n_samples: int,
    device_class: str,
) -> float:
    per_step = _PER_STEP_BASELINE_S[device_class] * (meta.params_millions / 100.0)
    steps = max(1, (n_samples // max(1, batch_size))) * epochs
    return (n_trials * steps * per_step) / 3600.0


def _classify_severity(estimate: float, budget: float) -> Severity:
    if estimate <= 0:
        return Severity.AMPLE
    if budget <= 0:
        return Severity.TIGHT
    ratio = estimate / budget
    if ratio >= _OVER_RATIO:
        return Severity.OVER
    if ratio >= _TIGHT_RATIO:
        return Severity.TIGHT
    return Severity.AMPLE


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
    transformer, classic = [], []
    for node_idx, node_type, entry in _walk_modules_indexed(search_space):
        bucket = classic if entry.get("module_name") in {"linear", "catboost"} else transformer
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
    mixed_precision = entry.get("dtype") in {"fp16", "bf16"}
    mode = _TRANSFORMER_TRAINING_MODE.get(module, "inference")
    batch_size = _max_int(entry.get("batch_size"), 32)
    epochs = _max_int(entry.get("num_train_epochs"), 1 if mode == "inference" else 10)
    seq_len = _max_int(entry.get("max_length"), _DEFAULT_SEQ_LEN)

    vram = _vram_for_transformer(meta, mode, mixed_precision, batch_size=batch_size, seq_len=seq_len)
    ram = _ram_for_module(meta, stats)

    driver_max_batch: int | None = None
    if hardware.vram_gb > 0:
        driver_max_batch = _max_fitting_batch_size(
            weight_vram_gb=_weights_vram_for_transformer(meta, mode),
            vram_budget_gb=hardware.vram_gb,
            per_sample_gb=_activations_gb_per_sample(
                meta, seq_len, mixed_precision=mixed_precision, is_training=mode != "inference"
            ),
        )

    time_h = _time_for_transformer(
        meta=meta,
        n_trials=n_trials,
        epochs=epochs,
        batch_size=batch_size,
        n_samples=stats.n_samples,
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


def _aggregate_disk(
    estimate: ResourceEstimate,
    seen_models: dict[str, ModelMeta],
    node_max_weights: dict[int, float],
    *,
    dump_modules: bool,
    n_trials: int,
) -> None:
    """Fold per-model download/cached sizes into ``estimate`` and apply dump-modules accounting."""
    for meta in seen_models.values():
        if meta.cached_locally:
            estimate.disk_cached_gb += meta.disk_gb
        else:
            estimate.disk_download_gb += meta.disk_gb
    if dump_modules:
        # Each trial selects one variant per node, so per-trial dumped weights
        # are bounded by the heaviest module in each node, summed across nodes.
        estimate.disk_dump_gb = sum(node_max_weights.values()) * n_trials


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
            msg += f" (= per-trial {estimate.vram_gb:.1f} GB × {n_jobs} parallel trials)"
        msg += f" vs available {hardware.vram_gb:.1f} GB"
        report.add("resource", _classify_severity(effective_vram, hardware.vram_gb), msg, metric="vram")

    report.add(
        "resource",
        _classify_severity(effective_ram, hardware.ram_gb),
        f"RAM ~{effective_ram:.1f} GB vs available {hardware.ram_gb:.1f} GB",
        metric="ram",
    )

    disk_total = estimate.disk_download_gb + estimate.disk_dump_gb
    disk_msg = f"Disk ~{estimate.disk_download_gb:.1f} GB to download"
    if estimate.disk_cached_gb > 0:
        disk_msg += f", {estimate.disk_cached_gb:.1f} GB already cached"
    if estimate.disk_dump_gb > 0:
        disk_msg += f", +{estimate.disk_dump_gb:.1f} GB during training (dump_modules=True)"
    disk_msg += f" vs {hardware.free_disk_gb:.0f} GB free"
    report.add("resource", _classify_severity(disk_total, hardware.free_disk_gb), disk_msg, metric="disk")

    if estimate.time_hours > 0:
        report.add(
            "resource",
            Severity.AMPLE,
            f"Time ~{estimate.time_hours:.1f} h (worst case, no HPO pruning)",
            metric="time",
        )


def _resource_phase(
    config: dict[str, Any],
    stats: DatasetStats,
    hardware: HardwareProfile,
    report: PreflightReport,
) -> None:
    cfg = _validated_config(config)
    n_trials = max(1, cfg.hpo_config.n_trials)
    n_jobs = max(1, cfg.hpo_config.n_jobs)

    if not hub_reachable():
        report.low_confidence = True
        report.notes.append("HF Hub unreachable — all model sizes are name-pattern heuristics.")

    seen_models: dict[str, ModelMeta] = {}
    global_embedder = (cfg.embedder_config or {}).get("model_name")
    if global_embedder:
        seen_models[global_embedder] = resolve_model(global_embedder)

    transformer_entries, classic_entries = _split_entries(cfg.search_space)

    # First pass: transformer modules (also populates seen_models for the classic pass).
    module_estimates: list[_ModuleEstimate] = []
    node_max_weights: dict[int, float] = {}
    for node_idx, node_type, entry in transformer_entries:
        module = entry.get("module_name", "?")
        model_names = _extract_model_names(entry)
        if not model_names and global_embedder and module in {"knn", "mlknn"}:
            model_names = [global_embedder]
        for name in model_names:
            meta = seen_models.setdefault(name, resolve_model(name))
            me = _estimate_transformer_model(
                meta=meta,
                entry=entry,
                node_type=node_type,
                module=module,
                name=name,
                stats=stats,
                hardware=hardware,
                n_trials=n_trials,
                refit_after=cfg.refit_after,
            )
            module_estimates.append(me)
            # Track heaviest weight per node so dump_modules is bounded by one
            # selected variant per node x n_trials, not the sum of all candidates.
            node_max_weights[node_idx] = max(node_max_weights.get(node_idx, 0.0), me.model_weights_gb)

    # Second pass: linear / catboost — cost depends on embedder_dim, not a checkpoint.
    embedder_meta = _largest_embedder(seen_models)
    embedder_dim = _embedder_dim(embedder_meta)
    for _, node_type, entry in classic_entries:
        me = _estimate_classic_entry(
            entry=entry,
            node_type=node_type,
            embedder_meta=embedder_meta,
            embedder_dim=embedder_dim,
            stats=stats,
            hardware=hardware,
            n_trials=n_trials,
            refit_after=cfg.refit_after,
        )
        if me is not None:
            module_estimates.append(me)

    estimate = ResourceEstimate(parallel_factor=n_jobs)
    for me in module_estimates:
        estimate.vram_gb = max(estimate.vram_gb, me.vram_gb)
        estimate.ram_gb = max(estimate.ram_gb, me.ram_gb)
        estimate.time_hours += me.time_hours
        estimate.drivers.append(me.driver)

    _aggregate_disk(estimate, seen_models, node_max_weights, dump_modules=cfg.dump_modules, n_trials=n_trials)

    report.resource = estimate
    _emit_resource_findings(report, estimate, hardware, n_jobs=n_jobs)


def _config_phase(
    config: dict[str, Any],
    hardware: HardwareProfile,
    report: PreflightReport,
) -> None:
    hpo = config.get("hpo_config") or {}
    n_jobs = int(hpo.get("n_jobs", 1))

    if n_jobs > 1 and hardware.accelerator in {"cuda", "mps"}:
        report.add(
            "config",
            Severity.TIGHT,
            f"hpo_config.n_jobs={n_jobs} on a single GPU multiplies VRAM demand by {n_jobs}×.",
        )

    uses_catboost_gpu = False
    for _, entry in _walk_modules(config.get("search_space") or []):
        if entry.get("module_name") == "catboost" and entry.get("task_type") == "GPU":
            uses_catboost_gpu = True
            break
    if uses_catboost_gpu and hardware.accelerator != "cuda":
        report.add(
            "config",
            Severity.TIGHT,
            "CatBoost task_type=GPU configured but no CUDA detected — will fall back to CPU.",
        )


def _data_phase(
    config: dict[str, Any],
    stats: DatasetStats,
    report: PreflightReport,
) -> None:
    # token-length truncation (heuristic — we use stats.p95_tokens vs configured max_length)
    p95 = stats.p95_tokens or int(stats.avg_tokens * 2.5)
    for _, entry in _walk_modules(config.get("search_space") or []):
        max_len_value = entry.get("max_length")
        if max_len_value is None:
            continue
        max_len = _max_int(max_len_value, 512)
        if p95 > max_len:
            severity = Severity.OVER if p95 > max_len * 1.5 else Severity.TIGHT
            module_name = entry.get("module_name", "?")
            report.add(
                "data",
                severity,
                f"Train tokens p95~{p95} exceeds {module_name}.max_length={max_len}; expect silent truncation.",
            )

    # rare class x linear-CV (LogisticRegressionCV cv=3 needs >=3 samples/class;
    # multilabel path uses one-vs-rest without CV so the failure can't occur there)
    has_linear = any(e.get("module_name") == "linear" for _, e in _walk_modules(config.get("search_space") or []))
    if has_linear and stats.rare_classes and not stats.multilabel:
        report.add(
            "data",
            Severity.OVER,
            (f"LogisticRegressionCV (cv=3) will fail: classes {stats.rare_classes[:5]} have <3 samples."),
        )

    # partial descriptions x description scorer
    description_modules = {"description_bi", "description_cross", "description_llm"}
    has_description = any(
        e.get("module_name") in description_modules for _, e in _walk_modules(config.get("search_space") or [])
    )
    if has_description and stats.has_descriptions is False:
        report.add(
            "data",
            Severity.OVER,
            "description scorer present but intent descriptions are missing — fill them in or drop the scorer.",
        )


def run_preflight(
    config: dict[str, Any],
    stats: DatasetStats,
    hardware: HardwareProfile,
    *,
    preset_name: str | None = None,
) -> PreflightReport:
    """Run all three phases and return one report.

    Args:
        config: parsed preset / OptimizationConfig dict (top-level keys:
            ``search_space``, ``hpo_config``, optional ``embedder_config``).
        stats: dataset statistics (real or placeholder).
        hardware: detected hardware profile.
        preset_name: optional friendly name for the report header.

    Returns:
        PreflightReport with findings across resource/data/config phases.
    """
    report = PreflightReport(
        preset_name=preset_name,
        hardware={
            "accelerator": hardware.accelerator,
            "device_name": hardware.device_name,
            "vram_gb": round(hardware.vram_gb, 2),
            "ram_gb": round(hardware.ram_gb, 2),
            "free_disk_gb": round(hardware.free_disk_gb, 2),
            "device_class": hardware.device_class,
        },
        dataset={
            "n_samples": stats.n_samples,
            "n_classes": stats.n_classes,
            "avg_tokens": stats.avg_tokens,
            "p95_tokens": stats.p95_tokens,
            "multilabel": stats.multilabel,
            "source": stats.source,
        },
    )
    report.notes.extend(hardware.notes)

    _resource_phase(config, stats, hardware, report)
    _data_phase(config, stats, report)
    _config_phase(config, hardware, report)

    return report
