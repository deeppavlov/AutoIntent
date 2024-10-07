from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class RetrievalMetricFn(Protocol):
    def __call__(
        self,
        query_labels: list[int],
        candidates_labels: list[list[int]],
        k: int | None = None,
    ) -> float:
        """
        Arguments
        ---
        - `query_labels`: for each query, this list contains its class labels
        - `candidates_labels`: for each query, these lists contain class labels of items ranked by a retrieval model \
            (from most to least relevant)
        - `k`: the number of top items to consider for each query

        Note
        ---
        - multiclass case: labels are integer
        - multilabel case: labels are binary
        """
        ...


def macrofy(
    metric_fn: RetrievalMetricFn,
    query_labels: list[list[int]],
    candidates_labels: list[list[list[int]]],
    k: int | None = None,
) -> float:
    """
    extend single-label `metric_fn` to a multi-label case via macro averaging
    """
    query_labels_ = np.array(query_labels)
    candidates_labels_ = np.array(candidates_labels)

    n_classes = query_labels_.shape[1]
    classwise_values: list[float] = []
    for i in range(n_classes):
        binarized_query_labels = query_labels_[..., i]
        binarized_candidates_labels = candidates_labels_[..., i]
        classwise_values.append(metric_fn(binarized_query_labels, binarized_candidates_labels, k))

    return np.mean(classwise_values)  # type: ignore


def average_precision(query_label: int, candidate_labels: list[int], k: int | None = None) -> float:
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


def retrieval_map(query_labels: list[int], candidates_labels: list[list[int]], k: int | None = None) -> float:
    ap_list = [average_precision(q, c, k) for q, c in zip(query_labels, candidates_labels, strict=True)]
    return sum(ap_list) / len(ap_list)


def average_precision_intersecting(
    query_label: list[int], candidate_labels: list[list[int]], k: int | None = None
) -> float:
    """
    helper function for `retrieval_map_intersecting`
    """
    query_label_ = np.array(query_label)
    candidate_labels_ = np.array(candidate_labels)

    num_relevant = 0
    sum_precision = 0.0
    for i, label in enumerate(candidate_labels_[:k]):
        if np.sum(label * query_label_) > 0:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)
    return sum_precision / num_relevant if num_relevant > 0 else 0.0


def retrieval_map_intersecting(
    query_labels: list[list[int]],
    candidates_labels: list[list[list[int]]],
    k: int | None = None,
) -> float:
    ap_list = [average_precision_intersecting(q, c, k) for q, c in zip(query_labels, candidates_labels, strict=False)]
    return sum(ap_list) / len(ap_list)


def retrieval_map_macro(
    query_labels: list[list[int]],
    candidates_labels: list[list[list[int]]],
    k: int | None = None,
) -> float:
    return macrofy(retrieval_map, query_labels, candidates_labels, k)


def retrieval_map_numpy(query_labels: list[int], candidates_labels: list[list[int]], k: int) -> float:
    query_labels_ = np.array(query_labels)
    candidates_labels_ = np.array(candidates_labels)
    candidates_labels_ = candidates_labels_[:, :k]
    relevance_mask = candidates_labels == query_labels_[:, None]
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
    return np.mean(average_precision)  # type: ignore


def retrieval_hit_rate(query_labels: list[int], candidates_labels: list[list[int]], k: int | None = None) -> float:
    num_queries = len(query_labels)
    hit_count = 0

    for i in range(num_queries):
        query_label = query_labels[i]
        candidate_labels = candidates_labels[i][:k]

        if query_label in candidate_labels:
            hit_count += 1

    return hit_count / num_queries


def retrieval_hit_rate_intersecting(
    query_labels: list[list[int]], candidates_labels: list[list[list[int]]], k: int | None = None
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


def retrieval_hit_rate_macro(
    query_labels: list[list[int]],
    candidates_labels: list[list[list[int]]],
    k: int | None = None,
) -> float:
    return macrofy(retrieval_hit_rate, query_labels, candidates_labels, k)


def retrieval_hit_rate_numpy(query_labels: list[int], candidates_labels: list[list[int]], k: int) -> float:
    query_labels_ = np.array(query_labels)
    candidates_labels_ = np.array(candidates_labels)
    truncated_candidates = candidates_labels_[:, :k]
    hit_mask = np.isin(query_labels_[:, None], truncated_candidates).any(axis=1)
    return hit_mask.mean()  # type: ignore


def retrieval_precision(query_labels: list[int], candidates_labels: list[list[int]], k: int | None = None) -> float:
    total_precision = 0.0
    num_queries = len(query_labels)

    for i in range(num_queries):
        query_label = query_labels[i]
        candidate_labels = candidates_labels[i][:k]

        relevant_items = [label for label in candidate_labels if label == query_label]
        precision_at_k = len(relevant_items) / len(candidate_labels)

        total_precision += precision_at_k

    return total_precision / num_queries


def retrieval_precision_intersecting(
    query_labels: list[list[int]], candidates_labels: list[list[list[int]]], k: int | None = None
) -> float:
    total_precision = 0.0
    num_queries = len(query_labels)

    for i in range(num_queries):
        query_label = np.array(query_labels[i])  # (n_classes,)
        candidate_labels = np.array(candidates_labels[i][:k])  # (n_candidates, n_classes)

        relevant_items = [label for label in candidate_labels if np.sum(label * query_label) > 0]
        precision_at_k = len(relevant_items) / len(candidate_labels)

        total_precision += precision_at_k

    return total_precision / num_queries


def retrieval_precision_macro(
    query_labels: list[list[int]],
    candidates_labels: list[list[list[int]]],
    k: int | None = None,
) -> float:
    return macrofy(retrieval_precision, query_labels, candidates_labels, k)


def retrieval_precision_numpy(
    query_labels: list[int], candidates_labels: list[list[int]], k: int | None = None
) -> float:
    query_labels_ = np.array(query_labels)
    candidates_labels_ = np.array(candidates_labels)
    top_k_candidates = candidates_labels_[:, :k]
    matches = (top_k_candidates == query_labels_[:, None]).astype(int)
    relevant_counts = np.sum(matches, axis=1)
    precision_at_k = relevant_counts / k
    return np.mean(precision_at_k)  # type: ignore


def dcg(relevance_scores: npt.NDArray[Any], k: int | None = None) -> float:
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
    return np.sum(relevance_scores / discounts)  # type: ignore


def idcg(relevance_scores: npt.NDArray[Any], k: int | None = None) -> float:
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


def retrieval_ndcg(query_labels: list[int], candidates_labels: list[list[int]], k: int | None = None) -> float:
    ndcg_scores: list[float] = []
    relevance_scores = np.array(query_labels)[:, None] == np.array(candidates_labels)

    for rel_scores in relevance_scores:
        cur_dcg = dcg(rel_scores, k)
        cur_idcg = idcg(rel_scores, k)
        ndcg_scores.append(0.0 if cur_idcg == 0 else cur_dcg / cur_idcg)

    return sum(ndcg_scores) / len(ndcg_scores)


def retrieval_ndcg_intersecting(
    query_labels: list[list[int]], candidates_labels: list[list[list[int]]], k: int
) -> float:
    ndcg_scores: list[float] = []
    expanded_relevance_scores = np.array(query_labels)[:, None, :] == np.array(candidates_labels)
    relevance_scores = (expanded_relevance_scores.sum(axis=-1) != 0).astype(int)

    for rel_scores in relevance_scores:
        cur_dcg = dcg(rel_scores, k)
        cur_idcg = idcg(rel_scores, k)
        ndcg_scores.append(0.0 if cur_idcg == 0 else cur_dcg / cur_idcg)

    return sum(ndcg_scores) / len(ndcg_scores)


def retrieval_ndcg_macro(query_labels: list[list[int]], candidates_labels: list[list[list[int]]], k: int | None = None) -> float:
    return macrofy(retrieval_ndcg, query_labels, candidates_labels, k)


def retrieval_mrr(query_labels: list[int], candidates_labels: list[list[int]], k: int | None = None) -> float:
    mrr_sum = 0.0
    num_queries = len(query_labels)

    for i in range(num_queries):
        query_label = query_labels[i]
        candidate_labels = candidates_labels[i][:k]

        for rank, label in enumerate(candidate_labels):
            if label == query_label:
                mrr_sum += 1.0 / (rank + 1)
                break

    return mrr_sum / num_queries


def retrieval_mrr_intersecting(
    query_labels: list[int], candidates_labels: list[list[int]], k: int | None = None
) -> float:
    mrr_sum = 0.0
    num_queries = len(query_labels)

    for i in range(num_queries):
        query_label = np.array(query_labels[i])  # (n_classes,)
        candidate_labels = np.array(candidates_labels[i][:k])  # (n_candidates, n_classes)

        for rank, label in enumerate(candidate_labels):
            if np.sum(label * query_label) > 0:
                mrr_sum += 1.0 / (rank + 1)
                break

    return mrr_sum / num_queries


def retrieval_mrr_macro(query_labels: list[list[int]], candidates_labels: list[list[list[int]]], k: int | None = None) -> float:
    return macrofy(retrieval_mrr, query_labels, candidates_labels, k)
