def test_map():
    from autointent.metrics.retrieval import retrieval_map
    import numpy as np

    # case 1
    query_labels = [1]
    candidates_labels = [
        [2,1,1],
    ]
    k = None
    ground_truth = 7/12
    output = retrieval_map(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 2
    query_labels = [3]
    candidates_labels = [
        [3,1,1]
    ]
    k = None
    ground_truth = 1.
    output = retrieval_map(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 3
    query_labels = [1,3]
    candidates_labels = [
        [2,1,1],
        [3,1,1]
    ]
    k = None
    ground_truth = 19/24
    output = retrieval_map(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 4
    query_labels = [1,3]
    candidates_labels = [
        [2,1,1],
        [3,1,1]
    ]
    k = 2
    ground_truth = 0.75
    output = retrieval_map(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 5
    query_labels = [3]
    candidates_labels = [
        [2,1,1],
    ]
    k = None
    ground_truth = 0.
    output = retrieval_map(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 6
    query_labels = [3]
    candidates_labels = [
        [2,1,1],
    ]
    k = 2
    ground_truth = 0.
    output = retrieval_map(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 7
    query_labels = [3,1]
    candidates_labels = [
        [2,1,1],
        [2,4,4],
    ]
    k = None
    ground_truth = 0.
    output = retrieval_map(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

def test_hit_rate():
    from autointent.metrics.retrieval import retrieval_hit_rate
    import numpy as np

    # case 1
    query_labels = [1]
    candidates_labels = [
        [2,1,1],
    ]
    k = None

    ground_truth = 1.
    output = retrieval_hit_rate(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 2
    query_labels = [3]
    candidates_labels = [
        [2,1,1],
    ]
    k = None

    ground_truth = 0.
    output = retrieval_hit_rate(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 3
    query_labels = [1,3]
    candidates_labels = [
        [2,1,1],
        [3,1,1]
    ]
    k = None

    ground_truth = 1.
    output = retrieval_hit_rate(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 4
    query_labels = [1,3]
    candidates_labels = [
        [2,1,1],
        [3,1,1]
    ]
    k = 2

    ground_truth = 1.
    output = retrieval_hit_rate(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 5
    query_labels = [1,3]
    candidates_labels = [
        [2,1,1],
        [3,1,1]
    ]
    k = 1

    ground_truth = 0.5
    output = retrieval_hit_rate(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

def test_precision():
    from autointent.metrics.retrieval import retrieval_precision
    import numpy as np

    # case 1
    query_labels = [1]
    candidates_labels = [
        [2,1,1],
    ]
    k = None

    ground_truth = 2/3
    output = retrieval_precision(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 2
    query_labels = [3]
    candidates_labels = [
        [2,1,1],
    ]
    k = None

    ground_truth = 0.
    output = retrieval_precision(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 3
    query_labels = [1,3]
    candidates_labels = [
        [2,1,1],
        [3,1,1]
    ]
    k = None

    ground_truth = 0.5
    output = retrieval_precision(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 4
    query_labels = [1,3]
    candidates_labels = [
        [2,1,1],
        [3,1,1]
    ]
    k = 2

    ground_truth = 0.5
    output = retrieval_precision(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)


def test_ndcg():
    from autointent.metrics.retrieval import retrieval_ndcg
    import numpy as np

    # case 1
    query_labels = [1]
    candidates_labels = [
        [2,1,1],
    ]
    k = None

    dcg = 1 / np.log2(3) + 1 / np.log2(4)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    ground_truth = dcg / idcg
    output = retrieval_ndcg(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 2
    query_labels = [3]
    candidates_labels = [
        [2,1,1],
    ]
    k = None

    dcg = 0
    idcg = 0
    ground_truth = 0
    output = retrieval_ndcg(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 3
    query_labels = [1,3]
    candidates_labels = [
        [2,1,1],
        [3,1,1]
    ]
    k = None

    dcg_1 = 1 / np.log2(3) + 1 / np.log2(4)
    idcg_1 = 1 / np.log2(2) + 1 / np.log2(3)
    ground_truth_1 = dcg_1 / idcg_1
    dcg_2 = 1 / np.log2(2)
    idcg_2 = 1 / np.log2(2)
    ground_truth_2 = dcg_2 / idcg_2

    ground_truth = 0.5 * (ground_truth_1 + ground_truth_2)

    output = retrieval_ndcg(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 4
    query_labels = [1,3]
    candidates_labels = [
        [2,1,1],
        [3,1,1]
    ]
    k = 2

    dcg_1 = 1 / np.log2(3)
    idcg_1 = 1 / np.log2(2) + 1 / np.log2(3)
    ground_truth_1 = dcg_1 / idcg_1
    dcg_2 = 1 / np.log2(2)
    idcg_2 = 1 / np.log2(2)
    ground_truth_2 = dcg_2 / idcg_2

    ground_truth = 0.5 * (ground_truth_1 + ground_truth_2)

    output = retrieval_ndcg(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)


def test_mrr():
    from autointent.metrics.retrieval import retrieval_mrr
    import numpy as np

    # case 1
    query_labels = [1]
    candidates_labels = [
        [2,1,1],
    ]
    k = None

    ground_truth = 0.5
    output = retrieval_mrr(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 2
    query_labels = [3]
    candidates_labels = [
        [2,1,1],
    ]
    k = None

    ground_truth = 0.
    output = retrieval_mrr(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 3
    query_labels = [1,3]
    candidates_labels = [
        [2,1,1],
        [3,1,1]
    ]
    k = None

    ground_truth = 0.75
    output = retrieval_mrr(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)

    # case 4
    query_labels = [1,3]
    candidates_labels = [
        [2,2,1],
        [3,1,1]
    ]
    k = 2

    ground_truth = 0.5
    output = retrieval_mrr(query_labels, candidates_labels, k)
    np.testing.assert_almost_equal(output, ground_truth)