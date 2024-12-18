"""Retrieval node info."""

from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import (
    RETRIEVAL_METRICS_MULTICLASS,
    RETRIEVAL_METRICS_MULTILABEL,
    RetrievalMetricFn,
)
from autointent.modules import RETRIEVAL_MODULES_MULTICLASS, RETRIEVAL_MODULES_MULTILABEL
from autointent.modules.abc import Module

from ._base import NodeInfo


class EmbeddingNodeInfo(NodeInfo):
    """Retrieval node info."""

    metrics_available: ClassVar[Mapping[str, RetrievalMetricFn]] = (
        RETRIEVAL_METRICS_MULTICLASS | RETRIEVAL_METRICS_MULTILABEL
    )

    modules_available: ClassVar[Mapping[str, type[Module]]] = (
        RETRIEVAL_MODULES_MULTICLASS | RETRIEVAL_MODULES_MULTILABEL  # type: ignore[has-type]
    )

    node_type = NodeType.embedding
