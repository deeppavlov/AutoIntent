def test_knn_get_counts():
    import numpy as np
    from autointent.modules.scoring.knn.count_neighbors import get_counts

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
    weights = np.ones_like(labels)
    np.testing.assert_array_equal(x=get_counts(labels, n_classes, weights), y=ground_truth)

    # case 2
    labels = np.array([
        [1,2,1,1,2],
    ])
    n_classes = 3
    ground_truth = np.array([
        [0,3,2],
    ])
    weights = np.ones_like(labels)
    np.testing.assert_array_equal(x=get_counts(labels, n_classes, weights), y=ground_truth)

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
    weights = np.ones_like(labels)
    np.testing.assert_array_equal(x=get_counts(labels, n_classes, weights), y=ground_truth)


def test_scoring_get_topk():
    import numpy as np
    from autointent.modules.scoring.base import get_topk

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


def test_dnnc_build_result():
    import numpy as np
    from autointent.modules.scoring.dnnc import build_result

    scores = np.array([
        [0.0, 0.0, 0.2],
        [0.0, 0.3, 0.2],
        [0.3, 0.0, 0.2],
    ])
    labels = np.array([
        [4, 9, 3],
        [2, 5, 6],
        [7, 1, 0],
    ])
    n_classes = 10
    ground_truth = np.array([
        [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
    ])
    np.testing.assert_array_equal(
        x=build_result(scores, labels, n_classes),
        y=ground_truth
    )
