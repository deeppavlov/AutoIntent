"""Prediction node info."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import DECISION_METRICS, DICISION_METRICS_MULTILABEL
from autointent.modules import DECISION_MODULES

from ._base import NodeInfo

if TYPE_CHECKING:
    from collections.abc import Mapping

    from autointent.metrics import DecisionMetricFn
    from autointent.modules.base import BaseDecision


class DecisionNodeInfo(NodeInfo):
    """Prediction node info."""

    metrics_available: ClassVar[Mapping[str, DecisionMetricFn]] = DECISION_METRICS

    modules_available: ClassVar[dict[str, type[BaseDecision]]] = DECISION_MODULES

    node_type = NodeType.decision

    multiclass_available_metrics = DECISION_METRICS

    multilabel_available_metrics = DICISION_METRICS_MULTILABEL
