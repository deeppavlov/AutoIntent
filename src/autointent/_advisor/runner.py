"""Public entry point + config validation + data/config phases.

This file contains the central public function ``run_preflight`` at the top.
Everything below it is supporting machinery for the three phases.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from pydantic import ValidationError

from autointent._advisor._estimates._resource import _resource_phase
from autointent._advisor._estimates._search_space import _max_int, _module_cardinality, _walk_modules
from autointent._advisor._report import PreflightReport, Severity
from autointent._optimization_config import OptimizationConfig

if TYPE_CHECKING:
    from autointent._advisor._hardware import HardwareProfile
    from autointent._advisor._report import DatasetStats


logger = logging.getLogger(__name__)


def run_preflight(
    config: dict[str, Any],
    stats: DatasetStats,
    hardware: HardwareProfile,
    *,
    preset_name: str | None = None,
    refit_after: bool = False,
    embedding_cache_probe: Callable[[str], bool] | None = None,
) -> PreflightReport:
    """Run all three preflight phases and return one report.

    Args:
        config: parsed preset / ``OptimizationConfig`` dict (top-level keys:
            ``search_space``, ``hpo_config``, optional ``embedder_config``,
            optional ``logging_config.dump_modules``).
        stats: dataset statistics (real or placeholder).
        hardware: detected hardware profile.
        preset_name: optional friendly name for the report header.
        refit_after: matches the ``Pipeline.fit(refit_after=...)`` argument.
            When True, time estimates include the extra refit-on-full-data pass.
        embedding_cache_probe: optional callable ``(embedder_model_name) -> bool``.
            Return True when the embedding cache already holds this model's
            embeddings for the current dataset — the advisor then predicts 0
            forward time and 0 ``disk_embedding_cache_gb`` for that embedder
            (mirrors the ``cached_locally`` treatment for HF weights). Default
            is the pessimistic cold assumption every embedder pays once.

    Returns:
        ``PreflightReport`` with findings across resource / data / config phases.
    """
    cfg = _validated_config(config)
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

    _resource_phase(
        embedder_config=cfg.embedder_config,
        search_space=cfg.search_space,
        n_trials=cfg.hpo_config.n_trials,
        n_jobs=cfg.hpo_config.n_jobs,
        dump_modules=cfg.logging_config.dump_modules,
        stats=stats,
        hardware=hardware,
        report=report,
        refit_after=refit_after,
        cross_encoder_model_name=cfg.cross_encoder_config.model_name,
        transformer_model_name=cfg.transformer_config.model_name,
        cache_probe=embedding_cache_probe,
    )
    _data_phase(cfg.search_space, stats, report)
    _config_phase(cfg.search_space, cfg.hpo_config.n_jobs, cfg.hpo_config.n_trials, hardware, report)

    return report


def _validated_config(config: dict[str, Any]) -> OptimizationConfig:
    """Validate ``config`` against the project's canonical ``OptimizationConfig``.

    The advisor is best-effort: a malformed user config should still produce a
    report (with placeholder costs) rather than crashing, so any validation
    error falls back to the model defaults.
    """
    try:
        return OptimizationConfig.model_validate(config)
    except ValidationError as e:
        logger.warning("Advisor config failed validation; falling back to defaults: %s", e)
        # OptimizationConfig requires `search_space`; build a minimal valid default.
        return OptimizationConfig.model_validate({"search_space": []})


def _config_phase(
    search_space: list[dict[str, Any]],
    n_jobs: int,
    n_trials: int,
    hardware: HardwareProfile,
    report: PreflightReport,
) -> None:
    """Config-phase checks: parallelism vs. hardware mismatches + no-op HPO."""
    if n_jobs > 1 and hardware.accelerator in {"cuda", "mps"}:
        report.add(
            "config",
            Severity.TIGHT,
            f"hpo_config.n_jobs={n_jobs} on a single GPU multiplies VRAM demand by {n_jobs}x.",
        )

    uses_catboost_gpu = any(
        entry.get("module_name") == "catboost" and entry.get("task_type") == "GPU"
        for _, entry in _walk_modules(search_space)
    )
    if uses_catboost_gpu and hardware.accelerator != "cuda":
        report.add(
            "config",
            Severity.TIGHT,
            "CatBoost task_type=GPU configured but no CUDA detected - will fall back to CPU.",
        )

    # No-op HPO detector: n_trials >> search-space cardinality means most
    # trials will be exact duplicates of previous ones. Optuna's TPE sampler
    # doesn't auto-dedupe, so real runtime = n_trials × per-trial cost — the
    # advisor charges honestly for that. But the user probably didn't intend
    # this, so surface it as a finding: transformers-no-hpo on banking77
    # declares n_trials=40 with a single-value grid → 40x the useful work.
    for _, entry in _walk_modules(search_space):
        module = entry.get("module_name", "?")
        # Skip decision-node entries; they're cheap and often intentionally
        # singleton-configured.
        if module in {"argmax", "threshold", "jinoos", "tunable", "adaptive"}:
            continue
        cardinality = _module_cardinality(entry)
        if cardinality is not None and cardinality < n_trials and n_trials // max(1, cardinality) >= 4:
            report.add(
                "config",
                Severity.TIGHT,
                f"'{module}' entry has {cardinality} unique configurations but "
                f"hpo_config.n_trials={n_trials} — expect ~{n_trials - cardinality} "
                f"duplicate trials unless the sampler dedupes. Reduce n_trials or "
                f"widen the search space.",
            )
            # Emit at most one warning per preset — otherwise multi-module
            # presets with several singleton entries flood the report.
            break


def _data_phase(
    search_space: list[dict[str, Any]],
    stats: DatasetStats,
    report: PreflightReport,
) -> None:
    """Data-phase checks: token truncation, rare classes, missing intent descriptions."""
    # token-length truncation (heuristic — we use stats.p95_tokens vs configured max_length)
    p95 = stats.p95_tokens or int(stats.avg_tokens * 2.5)
    for _, entry in _walk_modules(search_space):
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

    # sklearn LogisticRegressionCV inner-CV failure: each class needs >= cv samples.
    # cv is configurable per linear entry (default 3); use the strictest one across
    # the search space. Multilabel uses LogisticRegression (no CV), so skip there.
    if not stats.multilabel and stats.class_counts:
        linear_cvs = [
            _max_int(e.get("cv"), 3) for _, e in _walk_modules(search_space) if e.get("module_name") == "linear"
        ]
        if linear_cvs:
            cv_max = max(linear_cvs)
            failing = sorted(name for name, count in stats.class_counts.items() if count < cv_max)
            if failing:
                report.add(
                    "data",
                    Severity.OVER,
                    f"LogisticRegressionCV (cv={cv_max}) will fail: classes {failing[:5]} have <{cv_max} samples.",
                )

    # partial descriptions x description scorer
    description_modules = {"description_bi", "description_cross", "description_llm"}
    has_description = any(e.get("module_name") in description_modules for _, e in _walk_modules(search_space))
    if has_description and stats.has_descriptions is False:
        report.add(
            "data",
            Severity.OVER,
            "description scorer present but intent descriptions are missing - fill them in or drop the scorer.",
        )
