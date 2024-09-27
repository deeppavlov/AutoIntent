import numpy as np
import pytest

from autointent.metrics.retrieval import (
    retrieval_hit_rate,
    retrieval_map,
    retrieval_mrr,
    retrieval_ndcg,
    retrieval_precision,
)


@pytest.mark.parametrize(
    "query_labels, candidates_labels, k, ground_truth",
    [
        ([1], [[2, 1, 1]], None, 7 / 12),
        ([3], [[3, 1, 1]], None, 1.0),
        ([1, 3], [[2, 1, 1], [3, 1, 1]], None, 19 / 24),
        ([1, 3], [[2, 1, 1], [3, 1, 1]], 2, 0.75),
        ([3], [[2, 1, 1]], None, 0.0),
        ([3], [[2, 1, 1]], 2, 0.0),
        ([3, 1], [[2, 1, 1], [2, 4, 4]], None, 0.0),
    ],
)
def test_map(query_labels, candidates_labels, k, ground_truth):
    output = retrieval_map(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)


@pytest.mark.parametrize(
    "query_labels, candidates_labels, k, ground_truth",
    [
        ([1], [[2, 1, 1]], None, 1.0),
        ([3], [[2, 1, 1]], None, 0.0),
        ([3], [[2, 1, 1]], 2, 0.0),
        ([3, 1], [[2, 1, 1], [3, 1, 1]], 1, 0.0),
        ([1, 3], [[2, 1, 1], [3, 1, 1]], None, 1.0),
        ([1, 3], [[2, 1, 1], [3, 1, 1]], 2, 1.0),
    ],
)
def test_hit_rate(query_labels, candidates_labels, k, ground_truth):
    output = retrieval_hit_rate(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)


@pytest.mark.parametrize(
    "query_labels, candidates_labels, k, ground_truth",
    [
        ([1], [[2, 1, 1]], None, 2 / 3),
        ([3], [[2, 1, 1]], None, 0.0),
        ([3], [[2, 1, 1]], 2, 0.0),
        ([3, 1], [[2, 1, 1], [3, 1, 1]], 1, 0),
        ([1, 3], [[2, 1, 1], [3, 1, 1]], None, 0.5),
        ([1, 3], [[2, 1, 1], [3, 1, 1]], 2, 0.5),
    ],
)
def test_precision(query_labels, candidates_labels, k, ground_truth):
    output = retrieval_precision(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)


@pytest.mark.parametrize(
    "query_labels, candidates_labels, k, ground_truth",
    [
        ([1], [[2, 1, 1]], None, 0.6934264036172708),
        ([3], [[2, 1, 1]], None, 0.0),
        ([3], [[2, 1, 1]], 2, 0.0),
        ([3, 1], [[2, 1, 1], [3, 1, 1]], 1, 0.0),
        ([1, 3], [[2, 1, 1], [3, 1, 1]], None, 0.8467132018086354),
        ([1, 3], [[2, 1, 1], [3, 1, 1]], 2, 0.6934264036172708),
    ],
)
def test_ndcg(query_labels, candidates_labels, k, ground_truth):
    output = retrieval_ndcg(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)


@pytest.mark.parametrize(
    "query_labels, candidates_labels, k, ground_truth",
    [
        ([1], [[2, 1, 1]], None, 0.5),
        ([3], [[2, 1, 1]], None, 0.0),
        ([3], [[2, 1, 1]], 2, 0.0),
        ([3, 1], [[2, 1, 1], [3, 1, 1]], 1, 0.0),
        ([1, 3], [[2, 1, 1], [3, 1, 1]], None, 0.75),
        ([1, 3], [[2, 1, 1], [3, 1, 1]], 2, 0.75),
    ],
)
def test_mrr(query_labels, candidates_labels, k, ground_truth):
    output = retrieval_mrr(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)
