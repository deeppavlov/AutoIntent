"""Retrieval node info."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from autointent.custom_types import NodeType
from autointent.metrics import (
    RETRIEVAL_METRICS_MULTICLASS,
    RETRIEVAL_METRICS_MULTILABEL,
    SCORING_METRICS_MULTICLASS,
    SCORING_METRICS_MULTILABEL,
)
from autointent.modules import EMBEDDING_MODULES

from ._base import NodeInfo

if TYPE_CHECKING:
    from collections.abc import Mapping

    from autointent.metrics import (
        RetrievalMetricFn,
        ScoringMetricFn,
    )
    from autointent.modules.base import BaseEmbedding


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
        "Mapping[str, RetrievalMetricFn | ScoringMetricFn]", RETRIEVAL_METRICS_MULTICLASS | SCORING_METRICS_MULTICLASS
    )

    multilabel_available_metrics: ClassVar[Mapping[str, RetrievalMetricFn | ScoringMetricFn]] = cast(
        "Mapping[str, RetrievalMetricFn | ScoringMetricFn]", RETRIEVAL_METRICS_MULTILABEL | SCORING_METRICS_MULTILABEL
    )
