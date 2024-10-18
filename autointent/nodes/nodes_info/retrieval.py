from collections.abc import Mapping
from typing import ClassVar

from autointent.configs.modules import RETRIEVAL_MODULES_CONFIGS, ModuleConfig
from autointent.metrics import (
    RETRIEVAL_METRICS_MULTICLASS,
    RETRIEVAL_METRICS_MULTILABEL,
    RetrievalMetricFn,
)
from autointent.modules import RETRIEVAL_MODULES_MULTICLASS, RETRIEVAL_MODULES_MULTILABEL, Module

from .base import NodeInfo


class RetrievalNodeInfo(NodeInfo):
    metrics_available: ClassVar[Mapping[str, RetrievalMetricFn]] = (
        RETRIEVAL_METRICS_MULTICLASS | RETRIEVAL_METRICS_MULTILABEL
    )

    modules_available: ClassVar[Mapping[str, type[Module]]] = (
        RETRIEVAL_MODULES_MULTICLASS | RETRIEVAL_MODULES_MULTILABEL
    )

    modules_configs: ClassVar[Mapping[str, type[ModuleConfig]]] = RETRIEVAL_MODULES_CONFIGS

    node_type = "retrieval"
