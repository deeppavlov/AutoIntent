from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from skmultilearn.model_selection import IterativeStratification

if TYPE_CHECKING:
    import numpy.typing as npt

_MULTILABEL_NDIMS = 2
_RARE_LABEL_COUNT_SINGLETON = 1
_RARE_LABEL_COUNT_PAIR = 2
_COIN_FLIP_P = 0.5


def safe_multilabel_split_indices(
    y: npt.NDArray[Any], test_size: float, random_seed: int | None
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Split multilabel data with coverage guarantees for rare labels."""
    _validate_multilabel_matrix(y)
    n_samples = int(y.shape[0])
    rng = np.random.default_rng(random_seed)

    train_idx: set[int] = set()
    test_idx: set[int] = set()
    label_counts = y.sum(axis=0).astype(int)

    _force_singleton_labels(y=y, label_counts=label_counts, train_idx=train_idx)
    _force_pair_labels(y=y, label_counts=label_counts, train_idx=train_idx, test_idx=test_idx, rng=rng)

    forced = train_idx | test_idx
    remaining = np.array(sorted(set(range(n_samples)) - forced), dtype=int)
    _iterative_stratify_remaining(
        y=y,
        remaining=remaining,
        test_size=test_size,
        random_seed=random_seed,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    return _finalize_partition(n_samples=n_samples, train_idx=train_idx, test_idx=test_idx)


def _validate_multilabel_matrix(y: npt.NDArray[Any]) -> None:
    if y.ndim != _MULTILABEL_NDIMS:
        msg = (
            "Expected multilabel data to be a 2D matrix-like structure "
            f"(n_samples, n_labels), got shape={getattr(y, 'shape', None)!r}."
        )
        raise ValueError(msg)


def _assigned_split(sample_idx: int, train_idx: set[int], test_idx: set[int]) -> str | None:
    if sample_idx in train_idx:
        return "train"
    if sample_idx in test_idx:
        return "test"
    return None


def _force_singleton_labels(y: npt.NDArray[Any], label_counts: npt.NDArray[Any], train_idx: set[int]) -> None:
    for label, count in enumerate(label_counts):
        if int(count) != _RARE_LABEL_COUNT_SINGLETON:
            continue
        sample = int(np.flatnonzero(y[:, label])[0])
        train_idx.add(sample)


def _force_pair_samples(a: int, b: int, train_idx: set[int], test_idx: set[int], rng: np.random.Generator) -> None:
    a_split = _assigned_split(a, train_idx, test_idx)
    b_split = _assigned_split(b, train_idx, test_idx)

    if a_split is not None and b_split is None:
        (test_idx if a_split == "train" else train_idx).add(b)
        return
    if b_split is not None and a_split is None:
        (test_idx if b_split == "train" else train_idx).add(a)
        return
    if a_split is None and b_split is None:
        if rng.random() < _COIN_FLIP_P:
            train_idx.add(a)
            test_idx.add(b)
        else:
            train_idx.add(b)
            test_idx.add(a)


def _force_pair_labels(
    y: npt.NDArray[Any],
    label_counts: npt.NDArray[Any],
    train_idx: set[int],
    test_idx: set[int],
    rng: np.random.Generator,
) -> None:
    for label, count in enumerate(label_counts):
        if int(count) != _RARE_LABEL_COUNT_PAIR:
            continue
        samples = np.flatnonzero(y[:, label]).astype(int)
        a, b = sorted(samples.tolist(), key=lambda i: int(y[i].sum()))
        _force_pair_samples(a=a, b=b, train_idx=train_idx, test_idx=test_idx, rng=rng)


def _iterative_stratify_remaining(
    y: npt.NDArray[Any],
    remaining: npt.NDArray[Any],
    test_size: float,
    random_seed: int | None,
    train_idx: set[int],
    test_idx: set[int],
) -> None:
    if len(remaining) == 0:
        return
    if random_seed is not None:
        # Workaround for buggy nature of IterativeStratification from skmultilearn
        from transformers import set_seed

        set_seed(random_seed)
    splitter = IterativeStratification(
        n_splits=2,
        order=2,
        # NOTE: IterativeStratification expects fold distribution in (test, train) order,
        # but returns indices as (train, test). This matches the library's behavior and
        # keeps backward-compatible train/test sizes with prior implementation.
        sample_distribution_per_fold=[test_size, 1.0 - test_size],
    )
    train_r, test_r = next(splitter.split(np.arange(len(remaining)), y[remaining]))
    train_idx |= set(remaining[train_r].tolist())
    test_idx |= set(remaining[test_r].tolist())


def _finalize_partition(
    n_samples: int, train_idx: set[int], test_idx: set[int]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    train_arr = np.array(sorted(train_idx), dtype=int)
    test_arr = np.array(sorted(test_idx), dtype=int)

    if len(train_arr) + len(test_arr) != n_samples:
        msg = (
            "Multilabel split did not partition all samples: "
            f"n_samples={n_samples}, train={len(train_arr)}, test={len(test_arr)}."
        )
        raise RuntimeError(msg)
    if set(train_arr.tolist()) & set(test_arr.tolist()):
        msg = "Multilabel split produced overlapping train/test indices."
        raise RuntimeError(msg)
    return train_arr, test_arr
