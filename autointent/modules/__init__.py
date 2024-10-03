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

RETRIEVAL_MODULES_MULTICLASS: dict[str, type[RetrievalModule]] = {
    "vector_db": VectorDBModule,
}

RETRIEVAL_MODULES_MULTILABEL = RETRIEVAL_MODULES_MULTICLASS

SCORING_MODULES_MULTICLASS: dict[str, type[ScoringModule]] = {
    "dnnc": DNNCScorer,
    "knn": KNNScorer,
    "linear": LinearScorer,
}

SCORING_MODULES_MULTILABEL: dict[str, type[ScoringModule]] = {
    "knn": KNNScorer,
    "linear": LinearScorer,
    "mlknn": MLKnnScorer,
}

PREDICTION_MODULES_MULTICLASS: dict[str, type[PredictionModule]] = {
    "argmax": ArgmaxPredictor,
    "jinoos": JinoosPredictor,
    "threshold": ThresholdPredictor,
    "tunable": TunablePredictor,
}

PREDICTION_MODULES_MULTILABEL: dict[str, type[PredictionModule]] = {
    "threshold": ThresholdPredictor,
    "tunable": TunablePredictor,
}
