from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset

    from autointent import Dataset
    from autointent.configs import DataConfig

from ._safe_multilabel_stratification import _validate_multilabel_matrix
from ._stratification import StratifiedSplitter


@dataclass(frozen=True)
class SplitReadinessResult:
    """Result of checking whether a dataset can be stratified split.

    Attributes:
        ready: True if stratification can be performed (enough samples per class).
        underpopulated_classes: List of (label, count) for classes below the minimum.
        min_samples_per_class_required: Minimum samples per class used for the check.
        reason: Human-readable reason when not ready (e.g. OOS not configured).
    """

    ready: bool
    underpopulated_classes: list[tuple[int, int]]
    min_samples_per_class_required: int
    reason: str | None


def check_split_readiness(
    dataset: Dataset,
    split: str,
    config: DataConfig,
    allow_oos_in_train: bool | None = None,
) -> SplitReadinessResult:
    """Check whether the dataset has enough samples per class for stratified splitting.

    Uses the same OOS and stratification logic as :func:`split_dataset`, so downstream
    code can call this before creating a :class:`DataHandler` or calling :func:`split_dataset`
    and handle underpopulated classes (e.g. skip phase, log, or fail with a clear message).

    Args:
        dataset: The dataset to check (e.g. the same passed to :func:`split_dataset`).
        split: The split name to check (e.g. ``Split.TRAIN``).
        test_size: Proportion used for the test split (must match the value used when splitting).
        config: data config
        allow_oos_in_train: Same as in :func:`split_dataset`. If the dataset has OOS samples
            and this is not set, the function returns ``ready=False`` with a reason.

    Returns:
        SplitReadinessResult with ``ready``, ``underpopulated_classes``, and optional ``reason``.
    """
    min_samples_per_class = _min_samples_per_class_for_config(config=config)
    if split not in dataset:
        return SplitReadinessResult(
            ready=False,
            underpopulated_classes=[],
            min_samples_per_class_required=min_samples_per_class,
            reason=f"Dataset has no split '{split}'.",
        )
    hf_split = dataset[split]
    splitter = StratifiedSplitter(
        test_size=config.validation_size,
        label_feature=dataset.label_feature,
        random_seed=None,
    )
    inputs = splitter.get_stratify_inputs(hf_split, dataset.multilabel, allow_oos_in_train)
    if inputs.multilabel:
        underpopulated = _check_multilabel_counts(inputs.dataset, splitter.label_feature, min_samples_per_class)
        ready = len(underpopulated) == 0
        reason = None
        if not ready:
            parts = [f"label {label!r}: {count} (need {min_samples_per_class})" for label, count in underpopulated]
            reason = "Multilabel stratification requires at least {} positives per label. Underpopulated: {}.".format(
                min_samples_per_class, "; ".join(parts)
            )
        return SplitReadinessResult(
            ready=ready,
            underpopulated_classes=underpopulated,
            min_samples_per_class_required=min_samples_per_class,
            reason=reason,
        )
    underpopulated = _check_multiclass_counts(inputs.dataset, splitter.label_feature, min_samples_per_class)
    ready = len(underpopulated) == 0
    reason = None
    if not ready:
        parts = [f"class {label!r}: {count} (need {min_samples_per_class})" for label, count in underpopulated]
        reason = "Stratification requires at least {} samples per class. Underpopulated: {}.".format(
            min_samples_per_class, "; ".join(parts)
        )
    return SplitReadinessResult(
        ready=ready,
        underpopulated_classes=underpopulated,
        min_samples_per_class_required=min_samples_per_class,
        reason=reason,
    )


def _min_samples_per_class_for_config(config: DataConfig) -> int:
    """Return a recommended minimum samples-per-class for a given data config."""
    # Base requirement for a single stratified split.
    # For CV, the canonical lower bound is one example per fold.
    base = 2 if config.scheme == "ho" else int(config.n_folds)

    # separation_ratio triggers an extra stratified split of the effective train
    # pool (e.g. decision vs scoring), so we double the requirement.
    factor = 1 if config.separation_ratio is None else 2
    return base * factor


def _check_multiclass_counts(
    dataset: HFDataset, label_feature: str, min_samples_per_class: int
) -> list[tuple[int, int]]:
    """Return (label, count) for each class with fewer than min_samples_per_class samples."""
    labels: list[int] = dataset[label_feature]
    counts = Counter(labels)
    return [(label, count) for label, count in counts.items() if count < min_samples_per_class]


def _check_multilabel_counts(
    dataset: HFDataset, label_feature: str, min_samples_per_class: int
) -> list[tuple[int, int]]:
    """Return (label_idx, positive_count) for each label with fewer than min_samples_per_class positives."""
    y = np.asarray(dataset[label_feature])
    _validate_multilabel_matrix(y)
    counts = y.sum(axis=0).astype(int)
    return [(int(idx), int(count)) for idx, count in enumerate(counts) if count < min_samples_per_class]
