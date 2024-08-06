def test_neg_cross_entropy():
    import sys

    sys.path.append("/home/voorhs/repos/AutoIntent")

    from src.metrics.scoring import scoring_neg_cross_entropy
    import numpy as np

    # case 1
    labels = [0]
    scores = [
        [.1,.3,.5,.1],
    ]
    ground_truth = -np.log(.1)
    output = scoring_neg_cross_entropy(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 2
    labels = [0,1,2,3]
    scores = [
        [.1,.3,.5,.1],
        [.1,.3,.5,.1],
        [.1,.3,.5,.1],
        [.1,.3,.5,.1],
    ]
    ground_truth = -np.mean([np.log(.1),np.log(.3),np.log(.5),np.log(.1),])
    output = scoring_neg_cross_entropy(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)


def test_roc_auc():
    import sys

    sys.path.append("/home/voorhs/repos/AutoIntent")

    from src.metrics.scoring import scoring_roc_auc
    import numpy as np

    # case 1
    labels = [0,1,2,3]
    scores = [
        [.5,.3,.1,.1],
        [.1,.5,.3,.1],
        [.1,.3,.5,.1],
        [.1,.3,.1,.5],
    ]
    ground_truth = 1.
    output = scoring_roc_auc(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 2
    labels = [0,1,2,3]
    scores = [
        [.5,.3,.1,.1],
        [.4,.1,.3,.2],
        [.1,.3,.5,.1],
        [.1,.3,.1,.5],
    ]
    ground_truth = 0.75
    output = scoring_roc_auc(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 3
    labels = [0,1,2,3]
    scores = [
        [.5,.3,.1,.1],
        [.4,.2,.3,.1],
        [.1,.1,.5,.3],
        [.1,.3,.1,.5],
    ]
    ground_truth = 10/12
    output = scoring_roc_auc(labels, scores)
    np.testing.assert_almost_equal(output, ground_truth)
