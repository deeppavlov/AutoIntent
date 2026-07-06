"""High-level advisor workflows: ``inspect`` and ``recommend``.

Each workflow orchestrates the lower-level pieces (``load_config``,
``detect_hardware``, ``stats_from_dataset``, ``run_preflight``) into a single
typed call. They expose the same logic the CLI uses but accept Python
arguments instead of an ``argparse.Namespace`` — useful from notebooks,
integration tests, or any caller that wants a ``PreflightReport`` /
``RecommendationResult`` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import yaml
from datasets import ClassLabel, Sequence, load_dataset

from autointent.custom_types import SearchSpacePreset
from autointent.utils import load_preset

from ._hardware import detect_hardware
from ._report import DatasetStats, RecommendationResult, Severity
from .runner import run_preflight

if TYPE_CHECKING:
    from collections.abc import Iterable

    from autointent import Dataset

    from ._report import PreflightReport


logger = logging.getLogger("autointent.advisor")

_SAMPLE_LIMIT = 1000
_P95_PERCENTILE = 0.95
BUNDLED_PRESETS: tuple[str, ...] = get_args(SearchSpacePreset)


def load_config(target: str) -> tuple[dict[str, Any], str]:
    """Return ``(config_dict, friendly_name)`` for either a preset name or a YAML path."""
    path = Path(target)
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f), path.stem
    return load_preset(target), target  # type: ignore[arg-type]


def stats_from_dataset(path: str, *, multilabel: bool = False) -> DatasetStats:
    """Best-effort: load a dataset via HF ``datasets.load_dataset`` and derive advisor stats.

    Accepts a Hub repo id (``DeepPavlov/clinc150``) or a local file path
    (``.csv`` / ``.json`` / ``.jsonl`` / ``.parquet``) / dataset directory. Falls
    back to a placeholder on any loader error so callers stay best-effort.
    """
    # Anything not in this map (no suffix, unknown suffix) is treated as a Hub
    # repo id or a dataset directory and passed to load_dataset directly.
    file_builders = {".csv": "csv", ".tsv": "csv", ".json": "json", ".jsonl": "json", ".parquet": "parquet"}
    builder = file_builders.get(Path(path).suffix.lower())
    try:
        ds = load_dataset(builder, data_files=path) if builder else load_dataset(path)
    except (OSError, ValueError, FileNotFoundError) as e:
        logger.warning("Failed to load dataset %s: %s", path, e)
        return DatasetStats.placeholder(multilabel=multilabel)

    train = ds["train"] if "train" in ds else next(iter(ds.values()), None)
    if train is None:
        return DatasetStats.placeholder(multilabel=multilabel)

    cols = train.column_names
    utt_col = next(
        (c for c in ("utterance", "text", "sentence", "query", "input") if c in cols), cols[0] if cols else None
    )
    label_col = next((c for c in ("label", "labels", "intent", "target") if c in cols), None)

    detected_multilabel, n_classes = _label_shape(train, label_col, fallback_multilabel=multilabel)

    sample = train[:_SAMPLE_LIMIT] if len(train) > _SAMPLE_LIMIT else train[:]
    lengths = [len(str(s).split()) for s in (sample.get(utt_col, []) if utt_col else [])]
    avg_tokens = int(sum(lengths) / max(1, len(lengths))) if lengths else 32
    if lengths:
        sorted_lengths = sorted(lengths)
        idx = max(0, min(len(sorted_lengths) - 1, round((len(sorted_lengths) - 1) * _P95_PERCENTILE)))
        p95 = sorted_lengths[idx]
    else:
        p95 = avg_tokens * 2

    return DatasetStats(
        n_samples=len(train),
        n_classes=n_classes,
        avg_tokens=avg_tokens,
        p95_tokens=p95,
        multilabel=detected_multilabel,
        has_descriptions=None,
        class_counts=_class_counts(train, label_col, detected_multilabel, n_classes) if label_col else {},
        source=f"dataset:{path}",
    )


def stats_from_dataset_obj(dataset: Dataset) -> DatasetStats:
    """Build :class:`DatasetStats` straight from an in-memory ``Dataset``.

    Counterpart of :func:`stats_from_dataset` that skips HF ``load_dataset``
    and reads the train split + autointent-specific attributes (``n_classes``,
    ``multilabel``, ``has_descriptions``) directly.
    """
    from autointent.custom_types import Split

    train_key = Split.TRAIN if Split.TRAIN in dataset else f"{Split.TRAIN}_0"
    if train_key not in dataset:
        return DatasetStats.placeholder()
    train = dataset[train_key]
    utt_col = dataset.utterance_feature
    label_col = dataset.label_feature

    sample = train[:_SAMPLE_LIMIT] if len(train) > _SAMPLE_LIMIT else train[:]
    lengths = [len(str(s).split()) for s in sample.get(utt_col, [])]
    avg_tokens = int(sum(lengths) / max(1, len(lengths))) if lengths else 32
    if lengths:
        sorted_lengths = sorted(lengths)
        idx = max(0, min(len(sorted_lengths) - 1, round((len(sorted_lengths) - 1) * _P95_PERCENTILE)))
        p95 = sorted_lengths[idx]
    else:
        p95 = avg_tokens * 2

    return DatasetStats(
        n_samples=len(train),
        n_classes=dataset.n_classes,
        avg_tokens=avg_tokens,
        p95_tokens=p95,
        multilabel=dataset.multilabel,
        has_descriptions=dataset.has_descriptions,
        class_counts=_class_counts(train, label_col, dataset.multilabel, dataset.n_classes),
        source="dataset:in-memory",
    )


def _label_shape(train: Any, label_col: str | None, *, fallback_multilabel: bool) -> tuple[bool, int]:  # noqa: ANN401
    """Derive ``(multilabel, n_classes)`` from the HF feature schema with a value-based fallback."""
    if label_col is None:
        return fallback_multilabel, 0
    feature = train.features.get(label_col)
    if isinstance(feature, Sequence):
        inner = feature.feature
        if isinstance(inner, ClassLabel):
            return True, inner.num_classes
        # Sequence of plain ints — n_classes = max label index + 1.
        max_idx = max((max(row) for row in train[label_col] if row), default=-1)
        return True, max_idx + 1
    if isinstance(feature, ClassLabel):
        return False, feature.num_classes
    # Plain int/string column. Detect multilabel from the first non-empty row, then count uniques.
    is_multi = len(train) > 0 and isinstance(train[0][label_col], (list, tuple))
    if is_multi:
        max_idx = max((max(row) for row in train[label_col] if row), default=-1)
        return True, max_idx + 1
    return False, len({label for label in train[label_col] if label is not None})


def _class_counts(
    train: Any,  # noqa: ANN401
    label_col: str,
    multilabel: bool,
    n_classes: int,
) -> dict[str, int]:
    """Per-class sample counts in the train split; empty on any error."""
    try:
        labels = train[label_col]
    except (KeyError, AttributeError, TypeError):
        return {}
    counts: dict[str, int] = {}
    if multilabel:
        for row in labels:
            if not row:
                continue
            for i, v in enumerate(row):
                if v:
                    counts[str(i)] = counts.get(str(i), 0) + 1
        for i in range(n_classes):
            counts.setdefault(str(i), 0)
    else:
        for label in labels:
            counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def inspect(
    target: str,
    *,
    stats: DatasetStats | None = None,
    budget_vram_gb: float | None = None,
) -> PreflightReport:
    """Inspect a preset (or YAML config path) against the local hardware.

    Args:
        target: Bundled preset name (e.g. ``'transformers-light'``) or a YAML
            config path. The friendly name surfaced in the report is the file
            stem for paths and the preset name otherwise.
        stats: Dataset stats to score against. Defaults to a placeholder if
            ``None``.
        budget_vram_gb: Optional VRAM-budget override for the hardware probe.

    Returns:
        ``PreflightReport`` covering resource / data / config phases.
    """
    config, name = load_config(target)
    hardware = detect_hardware(vram_budget_gb=budget_vram_gb)
    return run_preflight(config, stats or DatasetStats.placeholder(), hardware, preset_name=name)


def recommend(
    *,
    stats: DatasetStats | None = None,
    presets: Iterable[str] | None = None,
    budget_vram_gb: float | None = None,
    budget_time_h: float | None = None,
) -> RecommendationResult:
    """Walk bundled presets and return the best feasible fit plus all per-preset reports.

    Args:
        stats: Dataset stats to score against. Defaults to a placeholder if ``None``.
        presets: Override of the preset list (defaults to ``BUNDLED_PRESETS``).
        budget_vram_gb: Optional VRAM-budget override for the hardware probe.
        budget_time_h: Optional wall-time ceiling in hours; presets exceeding it
            get an extra ``Severity.OVER`` finding so they drop out of the
            feasible ranking.

    Returns:
        ``RecommendationResult`` with the chosen preset name and full results list.

    Note:
        Among feasible presets we pick the heaviest one that still fits the
        hardware budget — "use what you have" semantics. This is a *cost*
        ranking, not a quality ranking: a heavier preset is not strictly better
        and may overfit on small datasets where a classic-* preset would win on
        accuracy. Override ``presets=`` if you want a different ranking.
    """
    hardware = detect_hardware(vram_budget_gb=budget_vram_gb)
    stats = stats or DatasetStats.placeholder()
    preset_iter = list(presets) if presets is not None else BUNDLED_PRESETS

    results: list[tuple[str, PreflightReport]] = []
    for preset in preset_iter:
        try:
            cfg = load_preset(preset)  # type: ignore[arg-type]
        except (OSError, ValueError, KeyError) as e:
            logger.debug("Skipping preset %s: %s", preset, e)
            continue
        report = run_preflight(cfg, stats, hardware, preset_name=preset)
        if budget_time_h is not None and report.resource.time_hours > budget_time_h:
            report.add(
                "resource",
                Severity.OVER,
                f"Estimated time {report.resource.time_hours:.1f} h exceeds budget {budget_time_h} h.",
            )
        results.append((preset, report))

    cost_rank = {name: i for i, name in enumerate(BUNDLED_PRESETS)}
    feasible = [(name, r) for name, r in results if r.is_feasible]
    feasible.sort(key=lambda pair: (cost_rank.get(pair[0], len(BUNDLED_PRESETS)), pair[0]))
    chosen = feasible[0][0] if feasible else None

    return RecommendationResult(chosen=chosen, results=results)
