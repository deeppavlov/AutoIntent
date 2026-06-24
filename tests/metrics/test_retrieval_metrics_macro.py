"""Unit tests for macro-averaged retrieval metrics.

These cover the ``*_macro`` variants and the shared ``_macrofy`` helper, which
binarizes multilabel inputs per class and averages the single-label metric over
classes. Ground-truth constants are the deterministic outputs of the metric
functions (the same convention as the other retrieval-metric tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from autointent.metrics.retrieval import (
    retrieval_hit_rate_macro,
    retrieval_map_macro,
    retrieval_mrr_macro,
    retrieval_ndcg_macro,
    retrieval_precision_macro,
)

if TYPE_CHECKING:
    from autointent.custom_types import ListOfLabels
    from autointent.metrics.custom_types import CANDIDATE_TYPE

# A single multilabel query and a two-query batch reused across metrics.
Q_SINGLE = [[0, 1, 0]]
C_SINGLE = [[[0, 0, 1], [0, 1, 0], [0, 1, 0]]]
Q_BATCH = [[0, 1, 0], [0, 0, 1]]
C_BATCH = [[[0, 0, 1], [0, 1, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 0], [0, 1, 0]]]


@pytest.mark.parametrize(
    ("query_labels", "candidates_labels", "k", "ground_truth"),
    [
        (Q_SINGLE, C_SINGLE, None, 0.7222222222222222),
        (Q_BATCH, C_BATCH, None, 0.861111111111111),
        (Q_BATCH, C_BATCH, 2, 0.8333333333333334),
    ],
)
def test_map_macro(
    query_labels: ListOfLabels,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None,
    ground_truth: float,
) -> None:
    np.testing.assert_almost_equal(retrieval_map_macro(query_labels, candidates_labels, k), ground_truth)


@pytest.mark.parametrize(
    ("query_labels", "candidates_labels", "k", "ground_truth"),
    [
        (Q_SINGLE, C_SINGLE, None, 1.0),
        (Q_BATCH, C_BATCH, None, 1.0),
        (Q_BATCH, C_BATCH, 2, 1.0),
    ],
)
def test_hit_rate_macro(
    query_labels: ListOfLabels,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None,
    ground_truth: float,
) -> None:
    np.testing.assert_almost_equal(retrieval_hit_rate_macro(query_labels, candidates_labels, k), ground_truth)


@pytest.mark.parametrize(
    ("query_labels", "candidates_labels", "k", "ground_truth"),
    [
        (Q_SINGLE, C_SINGLE, None, 0.7777777777777777),
        (Q_BATCH, C_BATCH, None, 0.6666666666666666),
        (Q_BATCH, C_BATCH, 2, 0.6666666666666666),
    ],
)
def test_precision_macro(
    query_labels: ListOfLabels,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None,
    ground_truth: float,
) -> None:
    np.testing.assert_almost_equal(retrieval_precision_macro(query_labels, candidates_labels, k), ground_truth)


@pytest.mark.parametrize(
    ("query_labels", "candidates_labels", "k", "ground_truth"),
    [
        (Q_SINGLE, C_SINGLE, None, 0.7956176024115139),
        (Q_BATCH, C_BATCH, None, 0.8978088012057569),
        (Q_BATCH, C_BATCH, 2, 0.7956176024115139),
    ],
)
def test_ndcg_macro(
    query_labels: ListOfLabels,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None,
    ground_truth: float,
) -> None:
    np.testing.assert_almost_equal(retrieval_ndcg_macro(query_labels, candidates_labels, k), ground_truth)


@pytest.mark.parametrize(
    ("query_labels", "candidates_labels", "k", "ground_truth"),
    [
        (Q_SINGLE, C_SINGLE, None, 0.6666666666666666),
        (Q_BATCH, C_BATCH, None, 0.8333333333333334),
        (Q_BATCH, C_BATCH, 2, 0.8333333333333334),
    ],
)
def test_mrr_macro(
    query_labels: ListOfLabels,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None,
    ground_truth: float,
) -> None:
    np.testing.assert_almost_equal(retrieval_mrr_macro(query_labels, candidates_labels, k), ground_truth)
