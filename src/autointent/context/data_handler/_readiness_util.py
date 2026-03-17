from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset

    from autointent import Dataset
    from autointent.configs import DataConfig

from ._safe_multilabel_stratification import _validate_multilabel_matrix
from ._stratification import StratifiedSplitter


class ClassCount(NamedTuple):
    id: int
    """Class (intent) index."""

    n_samples: int
    """Number of samples from the class (intent)."""


@dataclass(frozen=True)
class SplitReadinessResult:
    """Result of checking whether a dataset can be fed to autointent pipeline.

    Attributes:
        ready: True if stratification can be performed (enough samples per class).
        underpopulated_classes: List of (label, n_samples) for classes below the minimum.
        min_samples_per_class_required: Minimum samples per class used for the check.
        reason: Human-readable reason when not ready (e.g. OOS not configured).
    """

    ready: bool
    underpopulated_classes: list[ClassCount]
    min_samples_per_class_required: int
    reason: str | None


def check_split_readiness(
    dataset: Dataset,
    split: str,
    config: DataConfig,
    allow_oos_in_train: bool | None = None,
) -> SplitReadinessResult:
    """Check whether the dataset has enough samples per class for autointent pipeline.

    Args:
        dataset: The dataset to check (e.g. the same passed to :func:`split_dataset`).
        split: The split name to check (e.g. ``Split.TRAIN``).
        config: data config
        allow_oos_in_train: Same as in :func:`split_dataset`. If the split contains OOS samples
            and this is ``None``, this function raises ``ValueError`` (mirrors splitting behavior).
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
    expected_n_classes = _expected_n_classes(dataset, inputs.dataset, splitter.label_feature)

    if inputs.multilabel:
        underpopulated = _find_underpopulated_multilabel(inputs.dataset, splitter.label_feature, min_samples_per_class)
    else:
        underpopulated = _find_underpopulated_multiclass(
            inputs.dataset,
            splitter.label_feature,
            min_samples_per_class,
            expected_n_classes=expected_n_classes,
        )
    ready = len(underpopulated) == 0
    reason: str | None = None

    if ready and (not inputs.multilabel):
        split_ok, split_reason = _check_multiclass_split_size_feasibility(
            dataset=inputs.dataset,
            label_feature=splitter.label_feature,
            test_size=inputs.test_size,
            expected_n_classes=expected_n_classes,
        )
        if not split_ok:
            ready = False
            reason = split_reason

    if not ready and reason is None:
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


def _find_underpopulated_multiclass(
    dataset: HFDataset, label_feature: str, min_samples_per_class: int, expected_n_classes: int
) -> list[ClassCount]:
    """Return (label, count) for each class with fewer than min_samples_per_class samples."""
    labels: list[int] = dataset[label_feature]
    counts = Counter(labels)

    # Ensure "missing" classes are treated as 0-count (underpopulated)
    result: list[ClassCount] = []
    for label in range(int(expected_n_classes)):
        n_samples = int(counts.get(label, 0))
        if n_samples < min_samples_per_class:
            result.append(ClassCount(id=int(label), n_samples=n_samples))
    return result


def _find_underpopulated_multilabel(
    dataset: HFDataset, label_feature: str, min_samples_per_class: int
) -> list[ClassCount]:
    """Return (label_idx, positive_count) for each label with fewer than min_samples_per_class positives."""
    y = np.asarray(dataset[label_feature])
    _validate_multilabel_matrix(y)
    counts = y.sum(axis=0).astype(int)
    return [
        ClassCount(id=int(idx), n_samples=int(n_samples))
        for idx, n_samples in enumerate(counts)
        if n_samples < min_samples_per_class
    ]


def _check_multiclass_split_size_feasibility(
    dataset: HFDataset, label_feature: str, test_size: float, expected_n_classes: int
) -> tuple[bool, str | None]:
    """Return whether stratified train/test sizes are feasible for multiclass splits.

    Even if each class has >=2 samples, sklearn stratified splitting can fail when
    the requested train/test sizes are too small to include all classes.
    """
    labels = dataset[label_feature]
    n_classes = expected_n_classes
    n_samples = len(labels)

    # Mirror sklearn's float test_size -> n_test calculation (ceil).
    n_test = int(np.ceil(float(test_size) * n_samples))
    n_train = n_samples - n_test

    if n_test <= 0 or n_train <= 0:
        return (
            False,
            f"Requested split sizes are invalid (n_samples={n_samples}, test_size={test_size}).",
        )
    if n_test < n_classes:
        return (
            False,
            f"Stratified split would allocate too few test samples (n_test={n_test}) "
            f"for the number of classes (n_classes={n_classes}).",
        )
    if n_train < n_classes:
        return (
            False,
            f"Stratified split would allocate too few train samples (n_train={n_train}) "
            f"for the number of classes (n_classes={n_classes}).",
        )
    return True, None


def _expected_n_classes(dataset: Dataset, prepared: HFDataset, label_feature: str) -> int:
    if dataset.multilabel:
        return len(prepared[label_feature][0])
    labels: list[int] = prepared[label_feature]
    max_seen = max(labels) if labels else -1
    return max(dataset.n_classes, int(max_seen) + 1)
