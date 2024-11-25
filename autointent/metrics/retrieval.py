"""Retrieval metrics."""

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from autointent.metrics.converter import transform

from .custom_types import CANDIDATE_TYPE, LABELS_VALUE_TYPE


class RetrievalMetricFn(Protocol):
    """Protocol for retrieval metrics."""

    def __call__(
        self,
        query_labels: LABELS_VALUE_TYPE,
        candidates_labels: CANDIDATE_TYPE,
        k: int | None = None,
    ) -> float:
        """
        Calculate retrieval metric.

        - multiclass case: labels are integer
        - multilabel case: labels are binary


        :param query_labels: For each query, this list contains its class labels
        :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
         (from most to least relevant)
        :param k: Number of top items to consider for each query
        :return: Score of the retrieval metric
        """
        ...


def macrofy(
    metric_fn: Callable[[npt.NDArray[Any], npt.NDArray[Any], int | None], float],
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    """
    Extend single-label `metric_fn` to a multi-label case via macro averaging.

    :param metric_fn: Metric function
    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_labels_, candidates_labels_ = transform(query_labels, candidates_labels)

    n_classes = query_labels_.shape[1]
    classwise_values: list[float] = []
    for i in range(n_classes):
        binarized_query_labels = query_labels_[..., i]
        binarized_candidates_labels = candidates_labels_[..., i]
        classwise_values.append(metric_fn(binarized_query_labels, binarized_candidates_labels, k))

    return np.mean(classwise_values)  # type: ignore[return-value]


def average_precision(query_label: int, candidate_labels: npt.NDArray[np.int64], k: int | None = None) -> float:
    """
    Calculate the average precision at position k.

    :param query_label: For each query, this list contains its class labels
    :param candidate_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    num_relevant = 0
    sum_precision = 0.0
    for i, label in enumerate(candidate_labels[:k]):
        if label == query_label:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)
    return sum_precision / num_relevant if num_relevant > 0 else 0.0


def retrieval_map(query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None) -> float:
    """
    Calculate the mean average precision at position k.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    ap_list = [average_precision(q, c, k) for q, c in zip(query_labels, candidates_labels, strict=True)]
    return sum(ap_list) / len(ap_list)


def average_precision_intersecting(
    query_label: LABELS_VALUE_TYPE, candidate_labels: CANDIDATE_TYPE, k: int | None = None,
) -> float:
    """
    Calculate the average precision at position k for the intersecting labels.

    :param query_label: For each query, this list contains its class labels
    :param candidate_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_label_, candidate_labels_ = transform(query_label, candidate_labels)

    num_relevant = 0
    sum_precision = 0.0
    for i, label in enumerate(candidate_labels_[:k]):
        if np.sum(label * query_label_) > 0:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)
    return sum_precision / num_relevant if num_relevant > 0 else 0.0


def retrieval_map_intersecting(
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    """
    Calculate the mean average precision at position k for the intersecting labels.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    ap_list = [average_precision_intersecting(q, c, k) for q, c in zip(query_labels, candidates_labels, strict=True)]
    return sum(ap_list) / len(ap_list)


def retrieval_map_macro(
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    """
    Calculate the mean average precision at position k for the intersecting labels.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    return macrofy(retrieval_map, query_labels, candidates_labels, k)


def retrieval_map_numpy(query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int) -> float:
    """
    Calculate mean average precision at position k.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_label_, candidates_labels_ = transform(query_labels, candidates_labels)
    candidates_labels_ = candidates_labels_[:, :k]
    relevance_mask = candidates_labels_ == query_label_[:, None]
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
    return np.mean(average_precision)  # type: ignore[no-any-return]


def retrieval_hit_rate(
    query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None,
) -> float:
    """
    Calculate the hit rate at position k.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_label_, candidates_labels_ = transform(query_labels, candidates_labels)
    candidates_labels_ = candidates_labels_[:, :k]

    num_queries = query_label_.shape[0]
    hit_count = 0

    for query_label, candidate_labels in zip(query_label_, candidates_labels_, strict=False):
        if query_label in candidate_labels:
            hit_count += 1

    return hit_count / num_queries  # type: ignore[no-any-return]


def retrieval_hit_rate_intersecting(
    query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None,
) -> float:
    """
    Calculate the hit rate at position k for the intersecting labels.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_label_, candidates_labels_ = transform(query_labels, candidates_labels)
    candidates_labels_ = candidates_labels_[:, :k]

    num_queries = query_label_.shape[0]
    hit_count = 0

    for query_label, candidate_labels in zip(query_label_, candidates_labels_, strict=False):
        candidate_labels_sum = np.sum(candidate_labels, axis=0)

        if np.sum(query_label * candidate_labels_sum) > 0:
            hit_count += 1

    return hit_count / num_queries  # type: ignore[no-any-return]


def retrieval_hit_rate_macro(
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    """
    Calculate the hit rate at position k for the intersecting labels.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    return macrofy(retrieval_hit_rate, query_labels, candidates_labels, k)


def retrieval_hit_rate_numpy(query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int) -> float:
    """
    Calculate the hit rate at position k.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_label_, candidates_labels_ = transform(query_labels, candidates_labels)
    truncated_candidates = candidates_labels_[:, :k]
    hit_mask = np.isin(query_label_[:, None], truncated_candidates).any(axis=1)
    return hit_mask.mean()  # type: ignore[no-any-return]


def retrieval_precision(
    query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None,
) -> float:
    """
    Calculate the precision at position k.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_label_, candidates_labels_ = transform(query_labels, candidates_labels)
    candidates_labels_ = candidates_labels_[:, :k]

    total_precision = 0.0
    num_queries = query_label_.shape[0]

    for query_label, candidate_labels in zip(query_label_, candidates_labels_, strict=False):
        relevant_items = [label for label in candidate_labels if label == query_label]
        precision_at_k = len(relevant_items) / candidate_labels.shape[0]

        total_precision += precision_at_k

    return total_precision / num_queries  # type: ignore[no-any-return]


def retrieval_precision_intersecting(
    query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None,
) -> float:
    """
    Calculate the precision at position k for the intersecting labels.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_label_, candidates_labels_ = transform(query_labels, candidates_labels)
    candidates_labels_ = candidates_labels_[:, :k]

    total_precision = 0.0
    num_queries = query_label_.shape[0]

    for query_label, candidate_labels in zip(query_label_, candidates_labels_, strict=False):
        # (n_classes,), (n_candidates, n_classes)

        relevant_items = [label for label in candidate_labels if np.sum(label * query_label) > 0]
        precision_at_k = len(relevant_items) / len(candidate_labels)

        total_precision += precision_at_k

    return total_precision / num_queries  # type: ignore[no-any-return]


def retrieval_precision_macro(
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    """
    Calculate the precision at position k for the intersecting labels.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    return macrofy(retrieval_precision, query_labels, candidates_labels, k)


def retrieval_precision_numpy(
    query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None,
) -> float:
    """
    Calculate the precision at position k.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_label_, candidates_labels_ = transform(query_labels, candidates_labels)
    top_k_candidates = candidates_labels_[:, :k]
    matches = (top_k_candidates == query_label_[:, None]).astype(int)
    relevant_counts = np.sum(matches, axis=1)
    precision_at_k = relevant_counts / k
    return np.mean(precision_at_k)  # type: ignore[no-any-return]


def dcg(relevance_scores: npt.NDArray[Any], k: int | None = None) -> float:
    """
    Calculate the Discounted Cumulative Gain (DCG) at position k.

    :param relevance_scores: numpy array of relevance scores for items
    :param k: the number of top items to consider
    :return: DCG value at position k
    """
    relevance_scores = relevance_scores[:k]
    discounts = np.log2(np.arange(2, relevance_scores.shape[0] + 2))
    return np.sum(relevance_scores / discounts)  # type: ignore[no-any-return]


def idcg(relevance_scores: npt.NDArray[Any], k: int | None = None) -> float:
    """
    Calculate the Ideal Discounted Cumulative Gain (IDCG) at position k.

    :param relevance_scores: `np.array` of relevance scores for items
    :param k: the number of top items to consider
    :return: IDCG value at position k
    """
    ideal_scores = np.sort(relevance_scores)[::-1]
    return dcg(ideal_scores, k)


def retrieval_ndcg(query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None) -> float:
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at position k.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_label_, candidates_labels_ = transform(query_labels, candidates_labels)

    ndcg_scores: list[float] = []
    relevance_scores: npt.NDArray[np.bool] = query_label_[:, None] == candidates_labels_

    for rel_scores in relevance_scores:
        cur_dcg = dcg(rel_scores, k)
        cur_idcg = idcg(rel_scores, k)
        ndcg_scores.append(0.0 if cur_idcg == 0 else cur_dcg / cur_idcg)

    return np.mean(ndcg_scores)  # type: ignore[return-value]


def retrieval_ndcg_intersecting(
    query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None,
) -> float:
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at position k for the intersecting labels.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_labels_, candidates_labels_ = transform(query_labels, candidates_labels)
    ndcg_scores: list[float] = []
    expanded_relevance_scores: npt.NDArray[np.bool] = query_labels_[:, None, :] == candidates_labels_
    relevance_scores = (expanded_relevance_scores.sum(axis=-1) != 0).astype(int)

    for rel_scores in relevance_scores:
        cur_dcg = dcg(rel_scores, k)
        cur_idcg = idcg(rel_scores, k)
        ndcg_scores.append(0.0 if cur_idcg == 0 else cur_dcg / cur_idcg)

    return np.mean(ndcg_scores)  # type: ignore[return-value]


def retrieval_ndcg_macro(
    query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None,
) -> float:
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at position k for the intersecting labels.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    return macrofy(retrieval_ndcg, query_labels, candidates_labels, k)


def retrieval_mrr(query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) at position k.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_labels_, candidates_labels_ = transform(query_labels, candidates_labels)
    candidates_labels_ = candidates_labels_[:, :k]

    mrr_sum = 0.0
    num_queries = query_labels_.shape[0]

    for query_label, candidate_labels in zip(query_labels_, candidates_labels_, strict=False):
        for rank, label in enumerate(candidate_labels):
            if label == query_label:
                mrr_sum += 1.0 / (rank + 1)
                break

    return mrr_sum / num_queries  # type: ignore[no-any-return]


def retrieval_mrr_intersecting(
    query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None,
) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) at position k for the intersecting labels.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    query_labels_, candidates_labels_ = transform(query_labels, candidates_labels)
    candidates_labels_ = candidates_labels_[:, :k]
    mrr_sum = 0.0
    num_queries = query_labels_.shape[0]

    for query_label, candidate_labels in zip(query_labels_, candidates_labels_, strict=False):
        for rank, label in enumerate(candidate_labels):
            if np.sum(label * query_label) > 0:
                mrr_sum += 1.0 / (rank + 1)
                break

    return mrr_sum / num_queries  # type: ignore[no-any-return]


def retrieval_mrr_macro(
    query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None,
) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) at position k for the intersecting labels.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    return macrofy(retrieval_mrr, query_labels, candidates_labels, k)
