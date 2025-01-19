import numpy as np
import pytest

from autointent.modules.scoring._dnnc import build_result
from autointent.modules.scoring._knn.count_neighbors import get_counts
from autointent.modules.scoring._knn.weighting import closest_weighting


@pytest.mark.parametrize(
    ("labels", "n_classes", "ground_truth"),
    [
        (
            np.array(
                [
                    [1, 2, 1, 1, 2],
                    [1, 2, 1, 2, 2],
                    [0, 2, 1, 2, 2],
                ],
            ),
            3,
            np.array(
                [
                    [0, 3, 2],
                    [0, 2, 3],
                    [1, 1, 3],
                ],
            ),
        ),
        (
            np.array(
                [
                    [1, 2, 1, 1, 2],
                ],
            ),
            3,
            np.array(
                [
                    [0, 3, 2],
                ],
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2],
                    [1, 1, 1, 1, 1],
                ],
            ),
            3,
            np.array(
                [
                    [5, 0, 0],
                    [0, 0, 5],
                    [0, 5, 0],
                ],
            ),
        ),
    ],
)
def test_knn_get_counts(labels, n_classes, ground_truth):
    weights = np.ones_like(labels)
    np.testing.assert_array_equal(actual=get_counts(labels, n_classes, weights), desired=ground_truth)


@pytest.mark.parametrize(
    ("scores", "labels", "n_classes", "ground_truth"),
    [
        (
            np.array(
                [
                    [0.0, 0.0, 0.2],
                    [0.0, 0.3, 0.2],
                    [0.3, 0.0, 0.2],
                ],
            ),
            np.array(
                [
                    [4, 9, 3],
                    [2, 5, 6],
                    [7, 1, 0],
                ],
            ),
            10,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
                ],
            ),
        ),
    ],
)
def test_dnnc_build_result(scores, labels, n_classes, ground_truth):
    np.testing.assert_array_equal(actual=build_result(scores, labels, n_classes), desired=ground_truth)


@pytest.mark.parametrize(
    ("labels", "distances", "multilabel", "n_classes", "ground_truth"),
    [
        (np.array([[0, 2]]), np.array([[0.5, 0.3]]), False, 3, [[0.75, 0, 0.85]]),
        (np.array([[0, 2, 0, 2]]), np.array([[0.5, 0.3, 0.1, 0.5]]), False, 3, [[0.95, 0, 0.85]]),
        (
            np.array(
                [
                    [0, 2, 0, 2],
                    [0, 2, 0, 2],
                ],
            ),
            np.array(
                [
                    [0.5, 0.3, 0.1, 0.5],
                    [0.5, 0.3, 0.1, 0.1],
                ],
            ),
            False,
            3,
            [
                [0.95, 0, 0.85],
                [0.95, 0, 0.95],
            ],
        ),
        (
            np.array([[[1, 0, 0], [0, 0, 1]]]),
            np.array([[0.5, 0.3]]),
            True,
            3,
            [
                [0.75, 0, 0.85],
            ],
        ),
        (
            np.array([[[1, 0, 1], [0, 0, 1]]]),
            np.array([[0.5, 0.3]]),
            True,
            3,
            [
                [0.75, 0, 0.85],
            ],
        ),
        (
            np.array([[[1, 0, 0], [1, 0, 1]]]),
            np.array([[0.5, 0.3]]),
            True,
            3,
            [
                [0.85, 0, 0.85],
            ],
        ),
    ],
)
def test_closest_weighting(labels, distances, multilabel, n_classes, ground_truth):
    np.testing.assert_array_equal(
        actual=closest_weighting(labels, distances, multilabel, n_classes),
        desired=ground_truth,
    )
