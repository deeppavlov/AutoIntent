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
from .scoring import DescriptionScorer, DNNCScorer, KNNScorer, LinearScorer, MLKnnScorer, ScoringModule

RETRIEVAL_MODULES_MULTICLASS: dict[str, type[Module]] = {
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

PREDICTION_MODULES_MULTICLASS: dict[str, type[Module]] = {
    "argmax": ArgmaxPredictor,
    "jinoos": JinoosPredictor,
    "threshold": ThresholdPredictor,
    "tunable": TunablePredictor,
}

PREDICTION_MODULES_MULTILABEL: dict[str, type[Module]] = {
    "threshold": ThresholdPredictor,
    "tunable": TunablePredictor,
}
__all__ = [
    "Module",
    "ArgmaxPredictor",
    "JinoosPredictor",
    "PredictionModule",
    "ThresholdPredictor",
    "TunablePredictor",
    "RegExp",
    "RetrievalModule",
    "VectorDBModule",
    "DNNCScorer",
    "KNNScorer",
    "LinearScorer",
    "MLKnnScorer",
    "DescriptionScorer",
    "ScoringModule",
    "RETRIEVAL_MODULES_MULTICLASS",
    "RETRIEVAL_MODULES_MULTILABEL",
    "SCORING_MODULES_MULTICLASS",
    "SCORING_MODULES_MULTILABEL",
    "PREDICTION_MODULES_MULTICLASS",
    "PREDICTION_MODULES_MULTILABEL",
]
