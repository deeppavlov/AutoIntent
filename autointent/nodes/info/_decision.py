"""Prediction node info."""

from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import DECISION_METRICS, DICISION_METRICS_MULTILABEL, DecisionMetricFn
from autointent.modules import DECISION_MODULES
from autointent.modules.base import BaseDecision

from ._base import NodeInfo


class DecisionNodeInfo(NodeInfo):
    """Prediction node info."""

    metrics_available: ClassVar[Mapping[str, DecisionMetricFn]] = DECISION_METRICS

    modules_available: ClassVar[dict[str, type[BaseDecision]]] = DECISION_MODULES

    node_type = NodeType.decision

    multiclass_available_metrics = DECISION_METRICS

    multilabel_available_metrics = DICISION_METRICS_MULTILABEL
