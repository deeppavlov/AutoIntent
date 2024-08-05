def test():
    import sys

    sys.path.append("/home/voorhs/repos/AutoIntent")

    import numpy as np
    from src.modules.scoring.base import get_topk

    # case 1
    scores = np.array([
        [0,1,2,3,4,5],
        [0,1,2,3,4,5],
        [0,1,2,3,4,5],
    ])
    k = 1
    ground_true = np.array([[5],[5],[5]])
    np.testing.assert_array_equal(x=get_topk(scores, k=k), y=ground_true)

    # case 2
    k = 3
    ground_true = np.array([[5,4,3],[5,4,3],[5,4,3]])
    np.testing.assert_array_equal(x=get_topk(scores, k=k), y=ground_true)

    # case 3
    scores = np.array([
        [0,1,2,3,4,5]
    ])
    k = 1
    ground_true = np.array([[5]])
    np.testing.assert_array_equal(x=get_topk(scores, k=k), y=ground_true)

    # case 4
    scores = np.array([
        [0,1,2,3,4,5]
    ])
    k = 3
    ground_true = np.array([[5,4,3]])
    np.testing.assert_array_equal(x=get_topk(scores, k=k), y=ground_true)