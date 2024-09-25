import numpy as np
from autointent.modules.scoring.knn.count_neighbors import get_counts
from autointent.modules.scoring.base import get_topk
from autointent.modules.scoring.dnnc import build_result
import pytest


@pytest.mark.parametrize(
    "labels, n_classes, ground_truth",
    [
        (
            np.array(
                [
                    [1, 2, 1, 1, 2],
                    [1, 2, 1, 2, 2],
                    [0, 2, 1, 2, 2],
                ]
            ),
            3,
            np.array(
                [
                    [0, 3, 2],
                    [0, 2, 3],
                    [1, 1, 3],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 2, 1, 1, 2],
                ]
            ),
            3,
            np.array(
                [
                    [0, 3, 2],
                ]
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2],
                    [1, 1, 1, 1, 1],
                ]
            ),
            3,
            np.array(
                [
                    [5, 0, 0],
                    [0, 0, 5],
                    [0, 5, 0],
                ]
            ),
        ),
    ],
)
def test_knn_get_counts(labels, n_classes, ground_truth):
    weights = np.ones_like(labels)
    np.testing.assert_array_equal(x=get_counts(labels, n_classes, weights), y=ground_truth)


@pytest.mark.parametrize(
    "scores, k, ground_truth",
    [
        (
            np.array(
                [
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                ]
            ),
            1,
            np.array([[5], [5], [5]]),
        ),
        (
            np.array(
                [
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                ]
            ),
            3,
            np.array([[5, 4, 3], [5, 4, 3], [5, 4, 3]]),
        ),
        (np.array([[0, 1, 2, 3, 4, 5]]), 1, np.array([[5]])),
        (np.array([[0, 1, 2, 3, 4, 5]]), 3, np.array([[5, 4, 3]])),
    ],
)
def test_scoring_get_topk(scores, k, ground_truth):
    np.testing.assert_array_equal(x=get_topk(scores, k=k), y=ground_truth)


@pytest.mark.parametrize(
    "scores, labels, n_classes, ground_truth",
    [
        (
            np.array(
                [
                    [0.0, 0.0, 0.2],
                    [0.0, 0.3, 0.2],
                    [0.3, 0.0, 0.2],
                ]
            ),
            np.array(
                [
                    [4, 9, 3],
                    [2, 5, 6],
                    [7, 1, 0],
                ]
            ),
            10,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
                ]
            ),
        )
    ],
)
def test_dnnc_build_result(scores, labels, n_classes, ground_truth):
    np.testing.assert_array_equal(x=build_result(scores, labels, n_classes), y=ground_truth)
