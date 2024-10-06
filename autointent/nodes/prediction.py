from collections.abc import Callable
from typing import ClassVar

from autointent.metrics import (
    prediction_accuracy,
    prediction_f1,
    prediction_precision,
    prediction_recall,
    prediction_roc_auc, PredictionMetricFn,
)
from autointent.modules import ArgmaxPredictor, JinoosPredictor, ThresholdPredictor, TunablePredictor, PredictionModule
from .base import Node


class PredictionNode(Node):
    metrics_available: ClassVar[dict[str, PredictionMetricFn]] = {
        "prediction_accuracy": prediction_accuracy,
        "prediction_precision": prediction_precision,
        "prediction_recall": prediction_recall,
        "prediction_f1": prediction_f1,
        "prediction_roc_auc": prediction_roc_auc,
    }

    modules_available: ClassVar[dict[str, type[PredictionModule]]] = {
        "threshold": ThresholdPredictor,
        "argmax": ArgmaxPredictor,
        "jinoos": JinoosPredictor,
        "tunable": TunablePredictor,
    }

    node_type = "prediction"
