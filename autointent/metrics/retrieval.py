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
    r"""
    Extend single-label `metric_fn` to a multi-label case via macro averaging.

    The macro-average score is calculated as:

    .. math::

        \text{MacroAvg} = \frac{1}{C} \sum_{i=1}^{C} \text{metric}(y_{\text{true},i}, y_{\text{pred},i}, k)

    where:
    - :math:`C` is the number of classes,
    - :math:`y_{\text{true},i}` is the true binary indicator for the :math:`i`-th class label,
    - :math:`y_{\text{pred},i}` is the predicted binary indicator for the :math:`i`-th class label,
    - :math:`k` is the number of top predictions to consider for each query,
    - :math:`\text{metric}(y_{\text{true},i}, y_{\text{pred},i}, k)`
    is the metric function applied to the top-k predictions for each class.

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
    r"""
    Calculate the average precision at position k.

    The average precision is calculated as:

    .. math::

        \text{AP} = \frac{1}{\text{num_relevant}} \sum_{i=1}^{k} \mathbb{1}(y_{\text{true},i} = 1)
        \cdot \frac{\text{num_relevant}}{i+1}

    where:
    - :math:`k` is the number of top items to consider for each query,
    - :math:`\text{num_relevant}` is the number of relevant items in the top-k ranking,
    - :math:`y_{\text{true},i}` is the true label (query label) for the :math:`i`-th ranked item,
    - :math:`\mathbb{1}(y_{\text{true},i} = 1)` is the indicator function that equals 1 if the
    :math:`i`-th item is relevant,
    - :math:`\frac{\text{num_relevant}}{i+1}` is the precision at rank :math:`i`.

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
    r"""
    Calculate the mean average precision at position k.

    The Mean Average Precision (MAP) is computed as the average of the average precision
    (AP) scores for all queries. The average precision for a single query is calculated using
    the :func:`average_precision` function, which computes the precision at each rank
    position considering the top-k retrieved items.

    MAP is given by:

    .. math::

        \text{MAP} = \frac{1}{Q} \sum_{q=1}^{Q} \text{AP}(q, c, k)

    where:
    - :math:`Q` is the total number of queries,
    - :math:`\text{AP}(q, c, k)` is the average precision for the :math:`q`-th query,
    calculated considering the true labels for that query :math:`q`, the ranked candidate
    labels :math:`c`, and the number `k` which determines the number of top items to consider.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    ap_list = [average_precision(q, c, k) for q, c in zip(query_labels, candidates_labels, strict=True)]
    return sum(ap_list) / len(ap_list)


def average_precision_intersecting(
    query_label: LABELS_VALUE_TYPE,
    candidate_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    r"""
    Calculate the average precision at position k for the intersecting labels.

    The average precision for intersecting labels is calculated as:

    .. math::

        \text{AP} = \frac{1}{\text{num_relevant}} \sum_{i=1}^{k} \mathbb{1}\left(\sum_{j=1}^{C}
        y_{\text{true},j}(q) \cdot y_{\text{pred},j}(i) > 0 \right) \cdot \frac{\text{num_relevant}}{i+1}

    where:
    - :math:`k` is the number of top items to consider for each query,
    - :math:`\text{num_relevant}` is the number of relevant items in the top-k ranking,
    - :math:`y_{\text{true},j}(q)` is the true binary label for the :math:`j`-th
    class of the query :math:`q`,
    - :math:`y_{\text{pred},j}(i)` is the predicted binary label for the :math:`j`-th class
    of the :math:`i`-th ranked item,
    - :math:`\mathbb{1}(\cdot)` is the indicator function that equals 1 if the sum of the
    element-wise product of true and predicted labels is greater than 0, and 0 otherwise,
    - :math:`\frac{\text{num_relevant}}{i+1}` is the precision at rank :math:`i`.

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
    r"""
    Calculate the mean average precision at position k for the intersecting labels.

    The Mean Average Precision (MAP) for intersecting labels is computed as
    the average of the average precision (AP) scores for all queries. The average
    precision for a single query is calculated using the :func:`average_precision_intersecting`
    function, which considers the intersecting true and predicted labels for the
    top-k retrieved items.

    MAP is given by:

    .. math::

        \text{MAP} = \frac{1}{Q} \sum_{q=1}^{Q} \text{AP}_{\text{intersecting}}(q, c, k)

    where:
    - :math:`Q` is the total number of queries,
    - :math:`\text{AP}_{\text{intersecting}}(q, c, k)` is the average precision for the
    :math:`q`-th query, calculated using the intersecting true labels (`q`),
    predicted labels (`c`), and the number of top items (`k`) to consider.

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
    r"""
    Calculate the mean average precision at position k for the intersecting labels.

    This function internally uses :func:`retrieval_map` to calculate the MAP for each query and then
    applies :func:`macrofy` to perform macro-averaging across multiple queries.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    return macrofy(retrieval_map, query_labels, candidates_labels, k)


def retrieval_map_numpy(query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int) -> float:
    r"""
    Calculate mean average precision at position k.

    The mean average precision (MAP) at position :math:`k` is calculated as follows:

    .. math::

        \text{AP}_q = \frac{1}{|R_q|} \sum_{i=1}^{k} P_q(i) \cdot \mathbb{1}(y_{\text{true},q} = y_{\text{pred},i})

        \text{MAP}@k = \frac{1}{|Q|} \sum_{q=1}^{Q} \text{AP}_q

    where:
    - :math:`\text{AP}_q` is the average precision for query :math:`q`,
    - :math:`P_q(i)` is the precision at the :math:`i`-th position for query :math:`q`,
    - :math:`\mathbb{1}(y_{\text{true},q} = y_{\text{pred},i})` is the indicator function that equals
    1 if the true label of the query matches the predicted label at position :math:`i` and 0 otherwise,
    - :math:`|R_q|` is the total number of relevant items for query :math:`q`,
    - :math:`|Q|` is the total number of queries.

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
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    r"""
    Calculate the hit rate at position k.

    The hit rate is calculated as:

    .. math::

        \text{Hit Rate} = \frac{\sum_{i=1}^N \mathbb{1}(y_{\text{query},i} \in y_{\text{candidates},i}^{(1:k)})}{N}

    where:
    - :math:`N` is the total number of queries,
    - :math:`y_{\text{query},i}` is the true label for the :math:`i`-th query,
    - :math:`y_{\text{candidates},i}^{(1:k)}` is the set of top-k predicted labels for the :math:`i`-th query,
    - :math:`\mathbb{1}(\text{condition})` is the indicator function that equals 1 if the condition
    is true and 0 otherwise.

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
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    r"""
    Calculate the hit rate at position k for the intersecting labels.

    The intersecting hit rate is calculated as:

    .. math::

        \text{Hit Rate}_{\text{intersecting}} = \frac{\sum_{i=1}^N \mathbb{1} \left( \sum_{j=1}^k
        \left( y_{\text{query},i} \cdot y_{\text{candidates},i,j} \right) > 0 \right)}{N}

    where:
    - :math:`N` is the total number of queries,
    - :math:`y_{\text{query},i}` is the one-hot encoded label vector for the :math:`i`-th query,
    - :math:`y_{\text{candidates},i,j}` is the one-hot encoded label vector of the :math:`j`-th
    candidate for the :math:`i`-th query,
    - :math:`k` is the number of top candidates considered,
    - :math:`\mathbb{1}(\text{condition})` is the indicator function that equals 1 if the condition
    is true and 0 otherwise.

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
    r"""
    Calculate the hit rate at position k for the intersecting labels.

    This function internally uses :func:`retrieval_hit_rate` to calculate the hit rate at position :math:`k`
    for each query and applies :func:`macrofy` to perform macro-averaging across multiple queries.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    return macrofy(retrieval_hit_rate, query_labels, candidates_labels, k)


def retrieval_hit_rate_numpy(query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int) -> float:
    r"""
    Calculate the hit rate at position k.

    The hit rate is calculated as:

    .. math::

        \text{Hit Rate} = \frac{\sum_{i=1}^N \mathbb{1}(y_{\text{query},i} \in y_{\text{candidates},i}^{(1:k)})}{N}

    where:
    - :math:`N` is the total number of queries,
    - :math:`y_{\text{query},i}` is the true label for the :math:`i`-th query,
    - :math:`y_{\text{candidates},i}^{(1:k)}` is the set of top-k predicted labels for the :math:`i`-th query,
    - :math:`\mathbb{1}(\text{condition})` is the indicator function that equals 1 if the condition
    is true and 0 otherwise.

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
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    r"""
    Calculate the precision at position k.

    Precision at position :math:`k` is calculated as:

    .. math::

        \text{Precision@k} = \frac{1}{N} \sum_{i=1}^N \frac{|y_{\text{query},i} \cap
        y_{\text{candidates},i}^{(1:k)}|}{k}

    where:
    - :math:`N` is the total number of queries,
    - :math:`y_{\text{query},i}` is the true label for the :math:`i`-th query,
    - :math:`y_{\text{candidates},i}^{(1:k)}` is the set of top-k predicted labels for the :math:`i`-th query.

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
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    r"""
    Calculate the precision at position k for the intersecting labels.

    Precision at position :math:`k` for intersecting labels is calculated as:

    .. math::

        \text{Precision@k}_{\text{intersecting}} = \frac{1}{N} \sum_{i=1}^N
        \frac{\sum_{j=1}^k \mathbb{1} \left( y_{\text{query},i} \cdot y_{\text{candidates},i,j} > 0 \right)}{k}

    where:
    - :math:`N` is the total number of queries,
    - :math:`y_{\text{query},i}` is the one-hot encoded label vector for the :math:`i`-th query,
    - :math:`y_{\text{candidates},i,j}` is the one-hot encoded label vector of the :math:`j`-th
    candidate for the :math:`i`-th query,
    - :math:`k` is the number of top candidates considered,
    - :math:`\mathbb{1}(\text{condition})` is the indicator function that equals 1 if the
    condition is true and 0 otherwise.

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
    r"""
    Calculate the precision at position k for the intersecting labels.

    This function internally uses :func:`retrieval_precision` to calculate the precision at position :math:`k`
    for each query and applies :func:`macrofy` to perform macro-averaging across multiple queries.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    return macrofy(retrieval_precision, query_labels, candidates_labels, k)


def retrieval_precision_numpy(
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    r"""
    Calculate the precision at position k.

    Precision at position :math:`k` is calculated as:

    .. math::

        \text{Precision@k} = \frac{1}{N} \sum_{i=1}^N \frac{\sum_{j=1}^k
        \mathbb{1}(y_{\text{query},i} = y_{\text{candidates},i,j})}{k}

    where:
    - :math:`N` is the total number of queries,
    - :math:`y_{\text{query},i}` is the true label for the :math:`i`-th query,
    - :math:`y_{\text{candidates},i,j}` is the :math:`j`-th predicted label for the :math:`i`-th query,
    - :math:`\mathbb{1}(\text{condition})` is the indicator function that equals 1 if the
    condition is true and 0 otherwise,
    - :math:`k` is the number of top candidates considered.

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
    r"""
    Calculate the Discounted Cumulative Gain (DCG) at position k.

    DCG is calculated as:

    .. math::

        \text{DCG@k} = \sum_{i=1}^k \frac{r_i}{\log_2(i + 1)}

    where:
    - :math:`r_i` is the relevance score of the item at rank :math:`i`,
    - :math:`k` is the number of top items considered.

    :param relevance_scores: numpy array of relevance scores for items
    :param k: the number of top items to consider
    :return: DCG value at position k
    """
    relevance_scores = relevance_scores[:k]
    discounts = np.log2(np.arange(2, relevance_scores.shape[0] + 2))
    return np.sum(relevance_scores / discounts)  # type: ignore[no-any-return]


def idcg(relevance_scores: npt.NDArray[Any], k: int | None = None) -> float:
    r"""
    Calculate the Ideal Discounted Cumulative Gain (IDCG) at position k.

    IDCG is the maximum possible DCG that can be achieved if the relevance
    scores are sorted in descending order. It is calculated as:

    .. math::

        \text{IDCG@k} = \sum_{i=1}^k \frac{r_i^{\text{ideal}}}{\log_2(i + 1)}

    where:
    - :math:`r_i^{\text{ideal}}` is the relevance score of the item at rank :math:`i` in the ideal (sorted) order,
    - :math:`k` is the number of top items considered.

    :param relevance_scores: `np.array` of relevance scores for items
    :param k: the number of top items to consider
    :return: IDCG value at position k
    """
    ideal_scores = np.sort(relevance_scores)[::-1]
    return dcg(ideal_scores, k)


def retrieval_ndcg(query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None) -> float:
    r"""
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at position k.

    NDCG at position :math:`k` is calculated as:

    .. math::

        \text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}

    where:
    - :math:`\text{DCG@k}` is the Discounted Cumulative Gain at position :math:`k`,
    - :math:`\text{IDCG@k}` is the Ideal Discounted Cumulative Gain at position :math:`k`.

    The NDCG value is normalized such that it is between 0 and 1, where 1 indicates the ideal ranking.

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
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    r"""
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at position k for the intersecting labels.

    NDCG at position :math:`k` for intersecting labels is calculated as:

    .. math::

        \text{NDCG@k}_{\text{intersecting}} = \frac{\text{DCG@k}_{\text{intersecting}}}
        {\text{IDCG@k}_{\text{intersecting}}}

    where:
    - :math:`\text{DCG@k}_{\text{intersecting}}` is the Discounted Cumulative Gain for the
    intersecting labels at position :math:`k`,
    - :math:`\text{IDCG@k}_{\text{intersecting}}` is the Ideal Discounted Cumulative Gain
    for the intersecting labels at position :math:`k`.

    Intersecting relevance is determined by checking whether the query labels overlap with
    the candidate labels.
    NDCG values are normalized between 0 and 1, where 1 indicates the ideal ranking.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked
    by a retrieval model
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
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    r"""
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at position k for the intersecting labels.

    This function calculates NDCG using :func:`retrieval_ndcg` and applies it to each
    query using :func:`macrofy` to compute the macro-averaged score.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    return macrofy(retrieval_ndcg, query_labels, candidates_labels, k)


def retrieval_mrr(query_labels: LABELS_VALUE_TYPE, candidates_labels: CANDIDATE_TYPE, k: int | None = None) -> float:
    r"""
    Calculate the Mean Reciprocal Rank (MRR) at position k.

    MRR is calculated as:

    .. math::

        \text{MRR@k} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}

    where:
    - :math:`\text{rank}_i` is the rank position of the first relevant item in the top-k results for query :math:`i`,
    - :math:`N` is the total number of queries.

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
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    r"""
    Calculate the Mean Reciprocal Rank (MRR) at position k for the intersecting labels.

    MRR is calculated as:

    .. math::

        \text{MRR@k}_{\text{intersecting}} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}

    where:
    - :math:`\text{rank}_i` is the rank position of the first relevant (intersecting) item in the top-k
    results for query :math:`i`,
    - :math:`N` is the total number of queries.

    Intersecting relevance is determined by checking whether the query label intersects with the candidate labels.

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
    query_labels: LABELS_VALUE_TYPE,
    candidates_labels: CANDIDATE_TYPE,
    k: int | None = None,
) -> float:
    r"""
    Calculate the Mean Reciprocal Rank (MRR) at position k for the intersecting labels.

    This function calculates MRR using :func:`retrieval_mrr` and applies it to each
    query using :func:`macrofy` to compute the macro-averaged score.

    :param query_labels: For each query, this list contains its class labels
    :param candidates_labels: For each query, these lists contain class labels of items ranked by a retrieval model
     (from most to least relevant)
    :param k: Number of top items to consider for each query
    :return: Score of the retrieval metric
    """
    return macrofy(retrieval_mrr, query_labels, candidates_labels, k)
