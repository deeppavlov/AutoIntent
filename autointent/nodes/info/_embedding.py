"""Retrieval node info."""

from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import (
    RETRIEVAL_METRICS_MULTICLASS,
    RETRIEVAL_METRICS_MULTILABEL,
    SCORING_METRICS_MULTICLASS,
    SCORING_METRICS_MULTILABEL,
    RetrievalMetricFn,
    ScoringMetricFn,
)
from autointent.modules import EMBEDDING_MODULES_MULTICLASS, EMBEDDING_MODULES_MULTILABEL
from autointent.modules.abc import BaseEmbedding

from ._base import NodeInfo


class EmbeddingNodeInfo(NodeInfo):
    """Retrieval node info."""

    metrics_available: ClassVar[Mapping[str, RetrievalMetricFn | ScoringMetricFn]] = (
        RETRIEVAL_METRICS_MULTICLASS
        | RETRIEVAL_METRICS_MULTILABEL
        | SCORING_METRICS_MULTILABEL
        | SCORING_METRICS_MULTICLASS
    )

    modules_available: ClassVar[Mapping[str, type[BaseEmbedding]]] = (
        EMBEDDING_MODULES_MULTICLASS | EMBEDDING_MODULES_MULTILABEL
    )

    node_type = NodeType.embedding
