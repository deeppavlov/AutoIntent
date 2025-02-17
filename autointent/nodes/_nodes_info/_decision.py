"""Prediction node info."""

from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import PREDICTION_METRICS, DecisionMetricFn
from autointent.modules import PREDICTION_MODULES_MULTICLASS, PREDICTION_MODULES_MULTILABEL
from autointent.modules.abc import DecisionModule

from ._base import NodeInfo


class DecisionNodeInfo(NodeInfo):
    """Prediction node info."""

    metrics_available: ClassVar[Mapping[str, DecisionMetricFn]] = PREDICTION_METRICS

    modules_available: ClassVar[dict[str, type[DecisionModule]]] = (
        PREDICTION_MODULES_MULTICLASS | PREDICTION_MODULES_MULTILABEL
    )

    node_type = NodeType.decision
