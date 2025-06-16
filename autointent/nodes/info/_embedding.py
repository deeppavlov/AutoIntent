"""Retrieval node info."""

from collections.abc import Mapping
from typing import ClassVar, cast

from autointent.custom_types import NodeType
from autointent.metrics import (
    RETRIEVAL_METRICS_MULTICLASS,
    RETRIEVAL_METRICS_MULTILABEL,
    SCORING_METRICS_MULTICLASS,
    SCORING_METRICS_MULTILABEL,
    RetrievalMetricFn,
    ScoringMetricFn,
)
from autointent.modules import EMBEDDING_MODULES
from autointent.modules.base import BaseEmbedding

from ._base import NodeInfo


class EmbeddingNodeInfo(NodeInfo):
    """Retrieval node info."""

    metrics_available: ClassVar[Mapping[str, RetrievalMetricFn | ScoringMetricFn]] = (
        RETRIEVAL_METRICS_MULTICLASS
        | RETRIEVAL_METRICS_MULTILABEL
        | SCORING_METRICS_MULTILABEL
        | SCORING_METRICS_MULTICLASS
    )

    modules_available: ClassVar[Mapping[str, type[BaseEmbedding]]] = EMBEDDING_MODULES

    node_type = NodeType.embedding

    multiclass_available_metrics: ClassVar[Mapping[str, RetrievalMetricFn | ScoringMetricFn]] = cast(
        Mapping[str, RetrievalMetricFn | ScoringMetricFn], RETRIEVAL_METRICS_MULTICLASS | SCORING_METRICS_MULTICLASS
    )

    multilabel_available_metrics: ClassVar[Mapping[str, RetrievalMetricFn | ScoringMetricFn]] = cast(
        Mapping[str, RetrievalMetricFn | ScoringMetricFn], RETRIEVAL_METRICS_MULTILABEL | SCORING_METRICS_MULTILABEL
    )
