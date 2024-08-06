from ..metrics import (
    prediction_accuracy,
    prediction_f1,
    prediction_precision,
    prediction_recall,
    prediction_roc_auc,
)
from ..modules import ThresholdPredictor, ArgmaxPredictor
from .base import Node


class PredictionNode(Node):
    metrics_available = {
        "prediction_accuracy": prediction_accuracy,
        "prediction_precision": prediction_precision,
        "prediction_recall": prediction_recall,
        "prediction_f1": prediction_f1,
        "prediction_roc_auc": prediction_roc_auc,
    }

    modules_available = {
        "threshold": ThresholdPredictor,
        "argmax": ArgmaxPredictor
    }
