from typing import TypeVar

from .base import Module
from .prediction import (
    AdaptivePredictor,
    ArgmaxPredictor,
    JinoosPredictor,
    PredictionModule,
    ThresholdPredictor,
    TunablePredictor,
)
from .regexp import RegExp
from .retrieval import RetrievalModule, VectorDBModule
from .scoring import DescriptionScorer, DNNCScorer, KNNScorer, LinearScorer, MLKnnScorer, ScoringModule, SklearnScorer

T = TypeVar("T", bound=Module)


def create_modules_dict(modules: list[type[T]]) -> dict[str, type[T]]:
    return {module.name: module for module in modules}


RETRIEVAL_MODULES_MULTICLASS: dict[str, type[Module]] = create_modules_dict([VectorDBModule])

RETRIEVAL_MODULES_MULTILABEL = RETRIEVAL_MODULES_MULTICLASS

SCORING_MODULES_MULTICLASS: dict[str, type[ScoringModule]] = create_modules_dict(
    [DNNCScorer, KNNScorer, LinearScorer, DescriptionScorer, SklearnScorer]
)

SCORING_MODULES_MULTILABEL: dict[str, type[ScoringModule]] = create_modules_dict(
    [MLKnnScorer, LinearScorer, DescriptionScorer, SklearnScorer]
)

PREDICTION_MODULES_MULTICLASS: dict[str, type[Module]] = create_modules_dict(
    [ArgmaxPredictor, JinoosPredictor, ThresholdPredictor, TunablePredictor]
)

PREDICTION_MODULES_MULTILABEL: dict[str, type[Module]] = create_modules_dict(
    [AdaptivePredictor, ThresholdPredictor, TunablePredictor]
)

__all__ = [
    "Module",
    "AdaptivePredictor" "ArgmaxPredictor",
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
    "SklearnScorer",
    "RETRIEVAL_MODULES_MULTICLASS",
    "RETRIEVAL_MODULES_MULTILABEL",
    "SCORING_MODULES_MULTICLASS",
    "SCORING_MODULES_MULTILABEL",
    "PREDICTION_MODULES_MULTICLASS",
    "PREDICTION_MODULES_MULTILABEL",
]
