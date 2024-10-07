from collections.abc import Callable
from typing import ClassVar

from autointent.metrics import PREDICTION_METRICS_MULTICLASS, PREDICTION_METRICS_MULTILABEL
from autointent.modules import PREDICTION_MODULES_MULTICLASS, PREDICTION_MODULES_MULTILABEL

from .base import NodeInfo


class PredictionNodeInfo(NodeInfo):
    metrics_available: ClassVar[dict[str, Callable]] = PREDICTION_METRICS_MULTICLASS | PREDICTION_METRICS_MULTILABEL

    modules_available: ClassVar[dict[str, Callable]] = PREDICTION_MODULES_MULTICLASS | PREDICTION_MODULES_MULTILABEL

    node_type = "prediction"
