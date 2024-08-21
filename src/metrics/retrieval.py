import numpy as np

common_docstring = """
    Arguments
    ---
    - `query_labels`: for each query, this list contains its class labels
    - `candidates_labels`: for each query, these lists contain class labels of items ranked by a retrieval model (from most to least relevant)
    - `k`: the number of top items to consider for each query

    Return
    ---
    retrieval metric, averaged over all queries
    

    TODO:
    - implement multilabel case, where query_labels: list[list[int]], i.e. each query has multiple intents
"""


def average_precision(
    query_label: int, candidate_labels: list[int], k: int = None
) -> float:
    """
    helper function for `retrieval_map`
    """
    num_relevant = 0
    sum_precision = 0.0
    for i, label in enumerate(candidate_labels[:k]):
        if label == query_label:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)
    return sum_precision / num_relevant if num_relevant > 0 else 0.0


def retrieval_map(
    query_labels: list[int], candidates_labels: list[list[int]], k: int = None
):
    ap_list = [
        average_precision(q, c, k) for q, c in zip(query_labels, candidates_labels)
    ]
    return sum(ap_list) / len(ap_list)


def retrieval_map_numpy(
    query_labels: list[int], candidates_labels: list[list[int]], k: int
) -> float:
    query_labels = np.array(query_labels)
    candidates_labels = np.array(candidates_labels)
    candidates_labels = candidates_labels[:, :k]
    relevance_mask = candidates_labels == query_labels[:, None]
    cumulative_relevant = np.cumsum(relevance_mask, axis=1)
    precision_at_k = cumulative_relevant * relevance_mask / np.arange(1, k + 1)
    sum_precision = np.sum(precision_at_k, axis=1)
    num_relevant = np.sum(relevance_mask, axis=1)
    average_precision = np.divide(
        sum_precision,
        num_relevant,
        out=np.zeros_like(sum_precision),
        where=num_relevant != 0,
    )
    return np.mean(average_precision)


def retrieval_hit_rate(
    query_labels: list[int], candidates_labels: list[list[int]], k: int = None
) -> float:
    num_queries = len(query_labels)
    hit_count = 0

    for i in range(num_queries):
        query_label = query_labels[i]
        candidate_labels = candidates_labels[i][:k]

        if query_label in candidate_labels:
            hit_count += 1

    return hit_count / num_queries


def retrieval_hit_rate_multilabel(
    query_labels: list[list[int]], candidates_labels: list[list[list[int]]], k: int = None
) -> float:
    """all the labels are binarized"""
    num_queries = len(query_labels)
    hit_count = 0

    for i in range(num_queries):
        query_label = np.array(query_labels[i])
        candidate_labels = np.sum(candidates_labels[i][:k], axis=0)

        if np.sum(query_label * candidate_labels) > 0:
            hit_count += 1

    return hit_count / num_queries


def retrieval_hit_rate_numpy(
    query_labels: list[int], candidates_labels: list[list[int]], k: int
) -> float:
    query_labels = np.array(query_labels)
    candidates_labels = np.array(candidates_labels)
    truncated_candidates = candidates_labels[:, :k]
    hit_mask = np.isin(query_labels[:, None], truncated_candidates).any(axis=1)
    hit_rate = hit_mask.mean()
    return hit_rate


def retrieval_precision(
    query_labels: list[int], candidates_labels: list[list[int]], k: int = None
) -> float:
    total_precision = 0.0
    num_queries = len(query_labels)

    for i in range(num_queries):
        query_label = query_labels[i]
        candidate_labels = candidates_labels[i][:k]

        relevant_items = [label for label in candidate_labels if label == query_label]
        precision_at_k = len(relevant_items) / len(candidate_labels)

        total_precision += precision_at_k

    return total_precision / num_queries


def retrieval_precision_numpy(
    query_labels: list[int], candidates_labels: list[list[int]], k: int
) -> float:
    query_labels = np.array(query_labels)
    candidates_labels = np.array(candidates_labels)
    top_k_candidates = candidates_labels[:, :k]
    matches = (top_k_candidates == query_labels[:, None]).astype(int)
    relevant_counts = np.sum(matches, axis=1)
    precision_at_k = relevant_counts / k
    return np.mean(precision_at_k)


def dcg(relevance_scores, k):
    """
    Calculate the Discounted Cumulative Gain (DCG) at position k.

    Arguments
    ---
    - `relevance_scores`: numpy array of relevance scores for items
    - `k`: the number of top items to consider

    Return
    ---
    DCG value at position k
    """
    relevance_scores = relevance_scores[:k]
    discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
    dcg = np.sum(relevance_scores / discounts)
    return dcg


def idcg(relevance_scores, k):
    """
    Calculate the Ideal Discounted Cumulative Gain (IDCG) at position k.

    Arguments
    ---
    - `relevance_scores`: numpy array of relevance scores for items
    - `k`: the number of top items to consider

    Return
    ---
    IDCG value at position k
    """
    ideal_scores = np.sort(relevance_scores)[::-1]
    return dcg(ideal_scores, k)


def retrieval_ndcg(query_labels, candidates_labels, k=None):
    ndcg_scores = []
    relevance_scores = np.array(query_labels)[:, None] == np.array(candidates_labels)

    for rel_scores in relevance_scores:
        cur_dcg = dcg(rel_scores, k)
        cur_idcg = idcg(rel_scores, k)
        ndcg_scores.append(0.0 if cur_idcg == 0 else cur_dcg / cur_idcg)

    return sum(ndcg_scores) / len(ndcg_scores)


def retrieval_mrr(query_labels: list[int], candidates_labels: list[list[int]], k: int = None) -> float:
    mrr_sum = 0.0
    num_queries = len(query_labels)

    for i in range(num_queries):
        query_label = query_labels[i]
        candidate_labels = candidates_labels[i][:k]

        for rank, label in enumerate(candidate_labels):
            if label == query_label:
                mrr_sum += 1.0 / (rank + 1)
                break

    mrr = mrr_sum / num_queries
    return mrr
