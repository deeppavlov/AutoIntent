from typing import TypeVar

from .abc import Module, PredictionModule, RetrievalModule, ScoringModule
from .prediction import (
    AdaptivePredictor,
    ArgmaxPredictor,
    JinoosPredictor,
    ThresholdPredictor,
    TunablePredictor,
)
from .retrieval import VectorDBModule
from .scoring import DescriptionScorer, DNNCScorer, KNNScorer, LinearScorer, MLKnnScorer, RerankScorer

T = TypeVar("T", bound=Module)


def _create_modules_dict(modules: list[type[T]]) -> dict[str, type[T]]:
    return {module.name: module for module in modules}


RETRIEVAL_MODULES_MULTICLASS: dict[str, type[RetrievalModule]] = _create_modules_dict([VectorDBModule])

RETRIEVAL_MODULES_MULTILABEL = RETRIEVAL_MODULES_MULTICLASS

SCORING_MODULES_MULTICLASS: dict[str, type[ScoringModule]] = _create_modules_dict(
    [DNNCScorer, KNNScorer, LinearScorer, DescriptionScorer, RerankScorer]
)

SCORING_MODULES_MULTILABEL: dict[str, type[ScoringModule]] = _create_modules_dict(
    [MLKnnScorer, LinearScorer, DescriptionScorer],
)

PREDICTION_MODULES_MULTICLASS: dict[str, type[PredictionModule]] = _create_modules_dict(
    [ArgmaxPredictor, JinoosPredictor, ThresholdPredictor, TunablePredictor],
)

PREDICTION_MODULES_MULTILABEL: dict[str, type[PredictionModule]] = _create_modules_dict(
    [AdaptivePredictor, ThresholdPredictor, TunablePredictor],
)

__all__ = []  # type: ignore[var-annotated]
