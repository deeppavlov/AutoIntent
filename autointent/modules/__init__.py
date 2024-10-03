from .base import Module
from .prediction import (
    ArgmaxPredictor,
    JinoosPredictor,
    PredictionModule,
    ThresholdPredictor,
    TunablePredictor,
)
from .regexp import RegExp
from .retrieval import RetrievalModule, VectorDBModule
from .scoring import DNNCScorer, KNNScorer, LinearScorer, MLKnnScorer, ScoringModule

RETRIEVAL_MODULES_MULTICLASS = {
    "vector_db": VectorDBModule,
}

RETRIEVAL_MODULES_MULTILABEL = RETRIEVAL_MODULES_MULTICLASS

SCORING_MODULES_MULTICLASS = {
    "dnnc": DNNCScorer,
    "knn": KNNScorer,
    "linear": LinearScorer,
}

SCORING_MODULES_MULTILABEL = {
    "knn": KNNScorer,
    "linear": LinearScorer,
    "mlknn": MLKnnScorer,
}

PREDICTION_MODULES_MULTICLASS = {
    "argmax": ArgmaxPredictor,
    "jinoos": JinoosPredictor,
    "threshold": ThresholdPredictor,
    "tunable": TunablePredictor,
}

PREDICTION_MODULES_MULTILABEL = {
    "threshold": ThresholdPredictor,
    "tunable": TunablePredictor,
}
