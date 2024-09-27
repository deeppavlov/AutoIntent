import numpy as np
import pytest

from autointent.metrics.scoring import scoring_hit_rate, scoring_log_likelihood, scoring_neg_coverage, scoring_roc_auc


@pytest.mark.parametrize(
    "labels, scores, ground_truth",
    [
        ([0], [[0.1, 0.3, 0.5, 0.1]], np.log(0.1)),
        (
            [0, 1, 2, 3],
            [[0.1, 0.3, 0.5, 0.1], [0.1, 0.3, 0.5, 0.1], [0.1, 0.3, 0.5, 0.1], [0.1, 0.3, 0.5, 0.1]],
            np.mean(
                [
                    np.log(0.1),
                    np.log(0.3),
                    np.log(0.5),
                    np.log(0.1),
                ]
            ),
        ),
        (
            [[1, 0, 0, 0]],
            [[0.1, 0.3, 0.5, 0.1]],
            np.mean([np.log(0.1), np.log(1 - 0.3), np.log(1 - 0.5), np.log(1 - 0.1)]),
        ),
    ],
)
def test_neg_cross_entropy(labels, scores, ground_truth):
    output = scoring_log_likelihood(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)


@pytest.mark.parametrize(
    "labels, scores, ground_truth",
    [
        (
            [0, 1, 2, 3],
            [
                [0.5, 0.3, 0.1, 0.1],
                [0.1, 0.5, 0.3, 0.1],
                [0.1, 0.3, 0.5, 0.1],
                [0.1, 0.3, 0.1, 0.5],
            ],
            1.0,
        ),
        (
            [0, 1, 2, 3],
            [
                [0.5, 0.3, 0.1, 0.1],
                [0.4, 0.1, 0.3, 0.2],
                [0.1, 0.3, 0.5, 0.1],
                [0.1, 0.3, 0.1, 0.5],
            ],
            0.75,
        ),
        (
            [0, 1, 2, 3],
            [
                [0.5, 0.3, 0.1, 0.1],
                [0.4, 0.2, 0.3, 0.1],
                [0.1, 0.1, 0.5, 0.3],
                [0.1, 0.3, 0.1, 0.5],
            ],
            10 / 12,
        ),
    ],
)
def test_roc_auc(labels, scores, ground_truth):
    output = scoring_roc_auc(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)


@pytest.mark.parametrize(
    ["labels", "scores", "ground_truth"],
    [
        (
            [
                [1, 0, 0, 0],
                [
                    0,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                ],
            ],
            [
                [0.5, 0.3, 0.2, 0.2],
                [0.1, 0.5, 0.3, 0.2],
                [0.1, 0.3, 0.5, 0.2],
                [0.2, 0.3, 0.1, 0.5],
            ],
            1.0,
        ),
        (
            [
                [1, 0, 0, 0],
                [
                    0,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                ],
            ],
            [
                [0.5, 0.3, 0.1, 0.2],
                [0.1, 0.5, 0.3, 0.2],
                [0.1, 0.5, 0.3, 0.2],
                [0.2, 0.3, 0.1, 0.5],
            ],
            0.75,
        ),
        (
            [
                [1, 0, 0, 0],
                [
                    0,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    1,
                    1,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                ],
            ],
            [
                [0.5, 0.3, 0.1, 0.2],
                [0.1, 0.5, 0.3, 0.2],
                [0.1, 0.5, 0.3, 0.2],
                [0.1, 0.3, 0.2, 0.5],
            ],
            1,
        ),
        (
            [
                [1, 0, 0, 0],
            ],
            [
                [0.5, 0.3, 0.1, 0.2],
            ],
            1,
        ),
        (
            [
                [0, 1, 0, 0],
            ],
            [
                [0.5, 0.3, 0.1, 0.2],
            ],
            0,
        ),
    ],
)
def test_hit_rate(labels, scores, ground_truth):
    output = scoring_hit_rate(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)


@pytest.mark.parametrize(
    ["labels", "scores", "ground_truth"],
    [
        (
            [
                [1, 0, 0, 0],
            ],
            [
                [0.5, 0.3, 0.2, 0.1],
            ],
            1,
        ),
        (
            [
                [1, 1, 0, 0],
            ],
            [
                [0.5, 0.3, 0.2, 0.1],
            ],
            2 / 3,
        ),
        (
            [
                [0, 1, 0, 0],
            ],
            [
                [0.5, 0.3, 0.2, 0.1],
            ],
            2 / 3,
        ),
        (
            [
                [0, 0, 0, 1],
            ],
            [
                [0.5, 0.3, 0.2, 0.1],
            ],
            0,
        ),
        (
            [
                [1, 1, 1, 1],
            ],
            [
                [0.5, 0.3, 0.2, 0.1],
            ],
            0,
        ),
        (
            [
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            [
                [0.5, 0.3, 0.2, 0.1],
                [0.5, 0.3, 0.2, 0.1],
            ],
            1 / 3,
        ),
    ],
)
def test_coverage(labels, scores, ground_truth):
    output = scoring_neg_coverage(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)
