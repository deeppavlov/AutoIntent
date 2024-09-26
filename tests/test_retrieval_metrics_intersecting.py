import numpy as np
import pytest

from autointent.metrics.retrieval import (
    retrieval_hit_rate_intersecting,
    retrieval_map_intersecting,
    retrieval_mrr_intersecting,
    retrieval_ndcg_intersecting,
    retrieval_precision_intersecting,
)


@pytest.mark.parametrize(
    "query_labels, candidates_labels, k, ground_truth",
    [
        ([[0, 1, 0]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 7 / 12),
        ([[1, 1, 0]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 7 / 12),
        ([[0, 0, 1]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 1.0),
        ([[1, 0, 1]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 1.0),
        ([[0, 1, 0], [0, 0, 1]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 19 / 24),
        ([[0, 1, 0], [0, 0, 1]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 0], [0, 1, 0]]], 2, 0.75),
        ([[1, 0, 1]], [[[0, 0, 0], [0, 1, 0], [0, 1, 0]]], None, 0.0),
        ([[1, 0, 1]], [[[0, 0, 0], [0, 1, 0], [0, 1, 0]]], 2, 0.0),
    ],
)
def test_map(query_labels, candidates_labels, k, ground_truth):
    output = retrieval_map_intersecting(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)


@pytest.mark.parametrize(
    "query_labels, candidates_labels, k, ground_truth",
    [
        ([[0, 1, 0]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 1.0),
        ([[1, 1, 0]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 1.0),
        ([[0, 0, 1]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 1.0),
        ([[1, 0, 1]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 1.0),
        ([[0, 1, 0], [0, 0, 1]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 1.0),
        ([[0, 1, 0], [0, 0, 1]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 0], [0, 1, 0]]], 2, 1.0),
        ([[1, 0, 1]], [[[0, 0, 0], [0, 1, 0], [0, 1, 0]]], None, 0.0),
        ([[1, 0, 1]], [[[0, 0, 0], [0, 1, 0], [0, 1, 0]]], 2, 0.0),
    ],
)
def test_hit_rate(query_labels, candidates_labels, k, ground_truth):
    output = retrieval_hit_rate_intersecting(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)


@pytest.mark.parametrize(
    "query_labels, candidates_labels, k, ground_truth",
    [
        ([[0, 0, 1]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 1 / 3),
        ([[0, 1, 1]], [[[0, 0, 1], [0, 1, 0], [0, 1, 0]]], None, 1),
        ([[0, 1, 1]], [[[1, 0, 0], [1, 0, 0], [1, 0, 0]]], None, 0),
        ([[1, 0, 1], [0, 1, 1]], [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 0], [0, 0, 1]]], None, 0.5),
        ([[1, 0, 1], [0, 1, 1]], [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 0], [0, 0, 1]]], 2, 0.25),
    ],
)
def test_precision(query_labels, candidates_labels, k, ground_truth):
    output = retrieval_precision_intersecting(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)


@pytest.mark.parametrize(
    "query_labels, candidates_labels, k, ground_truth",
    [
        (
            [[0, 1, 1]],
            [[[1, 0, 0], [0, 1, 0], [1, 0, 1]]],
            None,
            0.6934264036172708,
        ),
        ([[0, 1, 1]], [[[1, 0, 0], [1, 0, 0], [1, 0, 0]]], None, 0),
        (
            [[0, 1, 1], [0, 1, 1]],
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0], [1, 0, 0]]],
            None,
            0.8467132018086354,
        ),
        (
            [[0, 1, 1], [0, 1, 1]],
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0], [1, 0, 0]]],
            2,
            0.6934264036172708,
        ),
    ],
)
def test_ndcg(query_labels, candidates_labels, k, ground_truth):
    output = retrieval_ndcg_intersecting(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)


@pytest.mark.parametrize(
    "query_labels, candidates_labels, k, ground_truth",
    [
        ([[0, 1, 1]], [[[1, 0, 0], [0, 1, 0], [1, 0, 1]]], None, 0.5),
        ([[0, 1, 1]], [[[1, 0, 0], [1, 0, 0], [1, 0, 0]]], None, 0.0),
        (
            [[0, 1, 1], [0, 1, 1]],
            [
                [[1, 0, 0], [1, 1, 0], [1, 0, 1]],
                [[1, 0, 1], [1, 1, 0], [1, 0, 1]],
            ],
            None,
            0.75,
        ),
        (
            [
                [0, 0, 1],
                [0, 1, 1],
            ],
            [
                [[1, 0, 0], [1, 1, 0], [1, 0, 1]],
                [[1, 0, 1], [1, 1, 0], [1, 0, 1]],
            ],
            2,
            0.5,
        ),
    ],
)
def test_mrr(query_labels, candidates_labels, k, ground_truth):
    output = retrieval_mrr_intersecting(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)
