"""Resource-phase estimation: walk the search space and aggregate cost.

Implements an honest worst-case for the modules the proposal lists as
in-scope. Formulas are intentionally coarse — the advisor's contract is
"heuristic upper bound, not measurement". Time and VRAM are the noisiest;
treat them as ballparks, not budgets.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from ._hardware import HardwareProfile
from ._hub import ModelMeta, hub_reachable, resolve_model
from ._report import DatasetStats, PreflightReport, ResourceEstimate, Severity

logger = logging.getLogger(__name__)

# yellow / red thresholds as fraction of available budget
_YELLOW = 0.7
_RED = 1.0

# rough per-step seconds, keyed on device class. Scaled by params_millions / 100.
_PER_STEP_BASELINE_S = {
    "cpu": 0.5,
    "low-gpu": 0.04,
    "mid-gpu": 0.02,
    "high-gpu": 0.01,
    "apple-silicon": 0.08,
}

TRANSFORMER_SCORER_MODULES = {"bert", "lora", "ptuning", "dnnc"}

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
        for c in cfg:
            if isinstance(c, dict) and c.get("model_name"):
                candidates.append(c["model_name"])
    elif isinstance(cfg, dict) and cfg.get("model_name"):
        candidates.append(cfg["model_name"])
    embedder_cfg = module_entry.get("embedder_config")
    if isinstance(embedder_cfg, list):
        for c in embedder_cfg:
            if isinstance(c, dict) and c.get("model_name"):
                candidates.append(c["model_name"])
    elif isinstance(embedder_cfg, dict) and embedder_cfg.get("model_name"):
        candidates.append(embedder_cfg["model_name"])
    return candidates


def _max_int(value: Any, default: int) -> int:
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


def _walk_modules(search_space: list[dict[str, Any]]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield (node_type, module_entry) pairs."""
    for node in search_space or []:
        node_type = node.get("node_type", "?")
        for entry in node.get("search_space", []) or []:
            yield node_type, entry


def _walk_modules_indexed(
    search_space: list[dict[str, Any]],
) -> Iterable[tuple[int, str, dict[str, Any]]]:
    """Yield (node_index, node_type, module_entry) — index lets us bound per-node max cost."""
    for node_idx, node in enumerate(search_space or []):
        node_type = node.get("node_type", "?")
        for entry in node.get("search_space", []) or []:
            yield node_idx, node_type, entry


def _vram_for_transformer(meta: ModelMeta, mode: str, mixed_precision: bool) -> float:
    """VRAM in GB for one trial of a transformer-based module.

    Full fine-tune fp32: weights + grads + Adam (m, v) = 4W.
    Full fine-tune AMP: fp16 weights + fp16 grads + fp32 master copy + fp32 Adam = 3W.
    (Activations are not modeled separately.)
    """
    weights_gb = meta.weights_gb
    if mode == "inference":
        return weights_gb * 1.3
    if mode == "lora":
        return weights_gb * 1.3 + 0.5
    if mode == "reranker":
        return weights_gb * 1.5
    if mixed_precision:
        return weights_gb * 3.0
    return weights_gb * 4.0


def _ram_for_module(meta: ModelMeta, stats: DatasetStats) -> float:
    """RAM in GB. Loose upper bound."""
    return meta.weights_gb + (stats.n_samples * stats.avg_tokens * 4) / (1024**3)


def _embedder_dim(meta: ModelMeta | None) -> int:
    """Coarse hidden-size guess from parameter count.

    Concrete points: MiniLM (33M) ~384, BERT-base (110M) ~768, BERT-large (350M) ~1024.
    """
    if meta is None:
        return 768
    params = meta.params_millions
    if params >= 300:
        return 1024
    if params >= 100:
        return 768
    if params >= 50:
        return 512
    return 384


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
    return (data_bytes + histograms_bytes + trees_bytes) / (1024**3)


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
    if budget <= 0:
        return Severity.YELLOW
    ratio = estimate / budget
    if ratio >= _RED:
        return Severity.RED
    if ratio >= _YELLOW:
        return Severity.YELLOW
    return Severity.GREEN


def _resource_phase(  # noqa: PLR0912 - kept linear for clarity
    config: dict[str, Any],
    stats: DatasetStats,
    hardware: HardwareProfile,
    report: PreflightReport,
) -> None:
    hpo = config.get("hpo_config") or {}
    n_trials = int(hpo.get("n_trials", 1))
    n_jobs = int(hpo.get("n_jobs", 1))
    refit_after = bool(config.get("refit_after", False))
    dump_modules = bool(config.get("dump_modules", False))

    if not hub_reachable():
        report.low_confidence = True
        report.notes.append("HF Hub unreachable — all model sizes are name-pattern heuristics.")

    seen_models: dict[str, ModelMeta] = {}
    estimate = ResourceEstimate(parallel_factor=max(1, n_jobs))

    embedder_cfg = config.get("embedder_config") or {}
    global_embedder = embedder_cfg.get("model_name") if isinstance(embedder_cfg, dict) else None
    if global_embedder:
        seen_models[global_embedder] = resolve_model(global_embedder)

    # First pass: walk transformer-bearing modules (collects seen_models for embedder_dim lookup).
    transformer_entries: list[tuple[int, str, dict[str, Any]]] = []
    classic_entries: list[tuple[int, str, dict[str, Any]]] = []
    for node_idx, node_type, entry in _walk_modules_indexed(config.get("search_space") or []):
        module = entry.get("module_name", "?")
        if module in {"linear", "catboost"}:
            classic_entries.append((node_idx, node_type, entry))
        else:
            transformer_entries.append((node_idx, node_type, entry))

    # Track the heaviest module per node so dump_modules accounting is bounded by
    # "one selected variant per node × n_trials", not "sum of every candidate".
    node_max_weights: dict[int, float] = {}

    for node_idx, node_type, entry in transformer_entries:
        module = entry.get("module_name", "?")
        model_names = _extract_model_names(entry)
        if not model_names and global_embedder and module in {"knn", "mlknn"}:
            model_names = [global_embedder]

        for name in model_names:
            meta = seen_models.setdefault(name, resolve_model(name))

            mixed_precision = entry.get("dtype") in {"fp16", "bf16"}
            if module == "bert":
                mode = "full-finetune"
            elif module == "lora":
                mode = "lora"
            elif module == "dnnc":
                mode = "reranker"
            elif module == "ptuning":
                mode = "full-finetune"
            else:
                mode = "inference"

            batch_size = _max_int(entry.get("batch_size"), 32)
            epochs = _max_int(entry.get("num_train_epochs"), 1 if mode == "inference" else 10)

            vram = _vram_for_transformer(meta, mode, mixed_precision)
            ram = _ram_for_module(meta, stats)

            time_h = 0.0
            if mode != "inference":
                time_h = _time_for_transformer(
                    meta=meta,
                    n_trials=n_trials,
                    epochs=epochs,
                    batch_size=batch_size,
                    n_samples=stats.n_samples,
                    device_class=hardware.device_class,
                )
            if refit_after and mode != "inference":
                time_h *= 1 + 1.0 / max(1, n_trials)

            estimate.vram_gb = max(estimate.vram_gb, vram)
            estimate.ram_gb = max(estimate.ram_gb, ram)
            estimate.time_hours += time_h
            node_max_weights[node_idx] = max(node_max_weights.get(node_idx, 0.0), meta.weights_gb)
            estimate.drivers.append(
                {
                    "node_type": node_type,
                    "module": module,
                    "model": name,
                    "mode": mode,
                    "vram_gb": round(vram, 2),
                    "ram_gb": round(ram, 2),
                    "time_hours": round(time_h, 2),
                    "confidence": meta.confidence,
                }
            )

    # Second pass: linear / catboost — cost depends on embedder_dim, not a checkpoint.
    embedder_meta = _largest_embedder(seen_models)
    embedder_dim = _embedder_dim(embedder_meta)
    class_multiplier_classic = max(1, stats.n_classes) if stats.multilabel else 1
    for _node_idx, node_type, entry in classic_entries:
        module = entry.get("module_name", "?")
        if module == "linear":
            max_iter = _max_int(entry.get("max_iter"), 100)
            cv_multiplier = 1 if stats.multilabel else _LOGREG_CV_MULTIPLIER
            ram = _ram_for_linear(stats=stats, embedder_dim=embedder_dim)
            time_h = _time_for_linear(
                n_trials=n_trials,
                n_samples=stats.n_samples,
                embedder_dim=embedder_dim,
                max_iter=max_iter,
                cv_multiplier=cv_multiplier,
                class_multiplier=class_multiplier_classic,
            )
            if refit_after:
                time_h *= 1 + 1.0 / max(1, n_trials)
            vram = 0.0
            mode = "linear-cv" if cv_multiplier > 1 else "linear"
            confidence = embedder_meta.confidence if embedder_meta else "heuristic"
        elif module == "catboost":
            iterations = _max_int(entry.get("iterations"), 1000)
            depth = _max_int(entry.get("depth"), 6)
            on_gpu = entry.get("task_type") == "GPU" and hardware.accelerator == "cuda"
            # CatBoost's multiclass MultiClass loss already grows per-class trees.
            cb_class_mult = max(1, stats.n_classes)
            ram = _ram_for_catboost(
                stats=stats,
                n_features=embedder_dim,
                iterations=iterations,
                depth=depth,
            )
            time_h = _time_for_catboost(
                n_trials=n_trials,
                n_samples=stats.n_samples,
                n_features=embedder_dim,
                iterations=iterations,
                depth=depth,
                class_multiplier=cb_class_mult,
                on_gpu=on_gpu,
            )
            if refit_after:
                time_h *= 1 + 1.0 / max(1, n_trials)
            vram = ram if on_gpu else 0.0
            if on_gpu:
                ram = 0.0
            mode = "catboost-gpu" if on_gpu else "catboost"
            confidence = embedder_meta.confidence if embedder_meta else "heuristic"
        else:
            continue

        estimate.vram_gb = max(estimate.vram_gb, vram)
        estimate.ram_gb = max(estimate.ram_gb, ram)
        estimate.time_hours += time_h
        estimate.drivers.append(
            {
                "node_type": node_type,
                "module": module,
                "model": embedder_meta.name if embedder_meta else "(no embedder)",
                "mode": mode,
                "vram_gb": round(vram, 2),
                "ram_gb": round(ram, 2),
                "time_hours": round(time_h, 2),
                "confidence": confidence,
            }
        )

    for meta in seen_models.values():
        if meta.cached_locally:
            estimate.disk_cached_gb += meta.disk_gb
        else:
            estimate.disk_download_gb += meta.disk_gb

    if dump_modules:
        # Each trial selects one variant per node, so per-trial dumped weights
        # are bounded by the heaviest module in each node, summed across nodes.
        per_trial_dump_gb = sum(node_max_weights.values())
        estimate.disk_dump_gb = per_trial_dump_gb * n_trials

    if n_jobs > 1 and hardware.accelerator in {"cuda", "mps"}:
        effective_vram = estimate.vram_gb * n_jobs
    else:
        effective_vram = estimate.vram_gb

    report.resource = estimate

    # render findings
    vram_sev = _classify_severity(effective_vram, hardware.vram_gb)
    if hardware.accelerator == "cpu" and effective_vram > 0:
        report.add(
            "resource",
            Severity.YELLOW,
            f"No GPU detected; transformer modules will be very slow (worst case ~{estimate.time_hours:.1f} h).",
            metric="vram",
        )
    else:
        msg = f"VRAM ~{effective_vram:.1f} GB"
        if n_jobs > 1:
            msg += f" (= per-trial {estimate.vram_gb:.1f} GB × {n_jobs} parallel trials)"
        msg += f" vs available {hardware.vram_gb:.1f} GB"
        report.add("resource", vram_sev, msg, metric="vram")

    ram_sev = _classify_severity(estimate.ram_gb, hardware.ram_gb)
    report.add(
        "resource",
        ram_sev,
        f"RAM ~{estimate.ram_gb:.1f} GB vs available {hardware.ram_gb:.1f} GB",
        metric="ram",
    )

    disk_total = estimate.disk_download_gb + estimate.disk_dump_gb
    disk_sev = _classify_severity(disk_total, hardware.free_disk_gb)
    disk_msg = f"Disk ~{estimate.disk_download_gb:.1f} GB to download"
    if estimate.disk_cached_gb > 0:
        disk_msg += f", {estimate.disk_cached_gb:.1f} GB already cached"
    if estimate.disk_dump_gb > 0:
        disk_msg += f", +{estimate.disk_dump_gb:.1f} GB during training (dump_modules=True)"
    disk_msg += f" vs {hardware.free_disk_gb:.0f} GB free"
    report.add("resource", disk_sev, disk_msg, metric="disk")

    if estimate.time_hours > 0:
        time_msg = f"Time ~{estimate.time_hours:.1f} h (worst case, no HPO pruning)"
        report.add("resource", Severity.GREEN, time_msg, metric="time")


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
            Severity.YELLOW,
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
            Severity.YELLOW,
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
            severity = Severity.RED if p95 > max_len * 1.5 else Severity.YELLOW
            report.add(
                "data",
                severity,
                f"Train tokens p95~{p95} exceeds {entry.get('module_name', '?')}.max_length={max_len}; expect silent truncation.",
            )

    # rare class × linear-CV
    has_linear = any(e.get("module_name") == "linear" for _, e in _walk_modules(config.get("search_space") or []))
    if has_linear and stats.rare_classes:
        report.add(
            "data",
            Severity.RED,
            (f"LogisticRegressionCV (cv=3) will fail: classes {stats.rare_classes[:5]} have <3 samples."),
        )

    # partial descriptions × description scorer
    has_description = any(
        e.get("module_name") == "description" for _, e in _walk_modules(config.get("search_space") or [])
    )
    if has_description and stats.has_descriptions is False:
        report.add(
            "data",
            Severity.RED,
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
