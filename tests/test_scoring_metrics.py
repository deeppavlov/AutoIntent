def test_nll():
    from autointent.metrics.scoring import scoring_log_likelihood
    import numpy as np

    # case 1
    labels = [0]
    scores = [
        [0.1, 0.3, 0.5, 0.1],
    ]
    ground_truth = np.log(0.1)
    output = scoring_log_likelihood(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 2
    labels = [0, 1, 2, 3]
    scores = [
        [0.1, 0.3, 0.5, 0.1],
        [0.1, 0.3, 0.5, 0.1],
        [0.1, 0.3, 0.5, 0.1],
        [0.1, 0.3, 0.5, 0.1],
    ]
    ground_truth = np.mean(
        [
            np.log(0.1),
            np.log(0.3),
            np.log(0.5),
            np.log(0.1),
        ]
    )
    output = scoring_log_likelihood(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)


def test_roc_auc():
    from autointent.metrics.scoring import scoring_roc_auc
    import numpy as np

    # case 1
    labels = [0, 1, 2, 3]
    scores = [
        [0.5, 0.3, 0.1, 0.1],
        [0.1, 0.5, 0.3, 0.1],
        [0.1, 0.3, 0.5, 0.1],
        [0.1, 0.3, 0.1, 0.5],
    ]
    ground_truth = 1.0
    output = scoring_roc_auc(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 2
    labels = [0, 1, 2, 3]
    scores = [
        [0.5, 0.3, 0.1, 0.1],
        [0.4, 0.1, 0.3, 0.2],
        [0.1, 0.3, 0.5, 0.1],
        [0.1, 0.3, 0.1, 0.5],
    ]
    ground_truth = 0.75
    output = scoring_roc_auc(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 3
    labels = [0, 1, 2, 3]
    scores = [
        [0.5, 0.3, 0.1, 0.1],
        [0.4, 0.2, 0.3, 0.1],
        [0.1, 0.1, 0.5, 0.3],
        [0.1, 0.3, 0.1, 0.5],
    ]
    ground_truth = 10 / 12
    output = scoring_roc_auc(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)
