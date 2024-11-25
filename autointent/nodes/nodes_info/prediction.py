"""Prediction node info."""

from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import PREDICTION_METRICS_MULTICLASS, PREDICTION_METRICS_MULTILABEL, PredictionMetricFn
from autointent.modules import PREDICTION_MODULES_MULTICLASS, PREDICTION_MODULES_MULTILABEL, Module

from .base import NodeInfo


class PredictionNodeInfo(NodeInfo):
    """Prediction node info."""

    metrics_available: ClassVar[Mapping[str, PredictionMetricFn]] = (
        PREDICTION_METRICS_MULTICLASS | PREDICTION_METRICS_MULTILABEL
    )

    modules_available: ClassVar[dict[str, type[Module]]] = PREDICTION_MODULES_MULTICLASS | PREDICTION_MODULES_MULTILABEL

    node_type = NodeType.prediction
