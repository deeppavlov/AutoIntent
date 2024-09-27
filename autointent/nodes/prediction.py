from collections.abc import Callable
from typing import ClassVar

from autointent.metrics import (
    prediction_accuracy,
    prediction_f1,
    prediction_precision,
    prediction_recall,
    prediction_roc_auc,
)
from autointent.modules import ArgmaxPredictor, JinoosPredictor, ThresholdPredictor, TunablePredictor

from .base import Node


class PredictionNode(Node):
    metrics_available: ClassVar[dict[str, Callable]] = {
        "prediction_accuracy": prediction_accuracy,
        "prediction_precision": prediction_precision,
        "prediction_recall": prediction_recall,
        "prediction_f1": prediction_f1,
        "prediction_roc_auc": prediction_roc_auc,
    }

    modules_available: ClassVar[dict[str, Callable]] = {
        "threshold": ThresholdPredictor,
        "argmax": ArgmaxPredictor,
        "jinoos": JinoosPredictor,
        "tunable": TunablePredictor,
    }

    node_type = "prediction"
