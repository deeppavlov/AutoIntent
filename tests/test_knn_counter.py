def test():
    import sys

    sys.path.append("/home/voorhs/repos/AutoIntent")

    import numpy as np
    from src.modules.scoring.knn import get_counts

    # case 1
    labels = np.array([
        [1,2,1,1,2],
        [1,2,1,2,2],
        [0,2,1,2,2],
    ])
    n_classes = 3
    ground_truth = np.array([
        [0,3,2],
        [0,2,3],
        [1,1,3],
    ])
    np.testing.assert_array_equal(x=get_counts(labels, n_classes), y=ground_truth)

    # case 2
    labels = np.array([
        [1,2,1,1,2],
    ])
    n_classes = 3
    ground_truth = np.array([
        [0,3,2],
    ])
    np.testing.assert_array_equal(x=get_counts(labels, n_classes), y=ground_truth)

    # case 3
    labels = np.array([
        [0,0,0,0,0],
        [2,2,2,2,2],
        [1,1,1,1,1],
    ])
    n_classes = 3
    ground_truth = np.array([
        [5,0,0],
        [0,0,5],
        [0,5,0],
    ])
    np.testing.assert_array_equal(x=get_counts(labels, n_classes), y=ground_truth)