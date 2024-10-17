from collections.abc import Callable
from typing import ClassVar

from autointent.metrics import RETRIEVAL_METRICS_MULTICLASS, RETRIEVAL_METRICS_MULTILABEL
from autointent.modules import RETRIEVAL_MODULES_MULTICLASS, RETRIEVAL_MODULES_MULTILABEL, Module

from .base import NodeInfo


class RetrievalNodeInfo(NodeInfo):
    metrics_available: ClassVar[dict[str, Callable]] = RETRIEVAL_METRICS_MULTICLASS | RETRIEVAL_METRICS_MULTILABEL

    modules_available: ClassVar[dict[str, type[Module]]] = RETRIEVAL_MODULES_MULTICLASS | RETRIEVAL_MODULES_MULTILABEL

    node_type = "retrieval"
