"""Heart of the library with different intent classification methods implemented."""

from typing import TypeVar

from .abc import DecisionModule, EmbeddingModule, Module, ScoringModule
from .decision import (
    AdaptiveDecision,
    ArgmaxDecision,
    JinoosDecision,
    ThresholdDecision,
    TunableDecision,
)
from .embedding import RetrievalEmbedding
from .scoring import DescriptionScorer, DNNCScorer, KNNScorer, LinearScorer, MLKnnScorer, RerankScorer

T = TypeVar("T", bound=Module)


def _create_modules_dict(modules: list[type[T]]) -> dict[str, type[T]]:
    return {module.name: module for module in modules}


RETRIEVAL_MODULES_MULTICLASS: dict[str, type[EmbeddingModule]] = _create_modules_dict([RetrievalEmbedding])

RETRIEVAL_MODULES_MULTILABEL = RETRIEVAL_MODULES_MULTICLASS

SCORING_MODULES_MULTICLASS: dict[str, type[ScoringModule]] = _create_modules_dict(
    [DNNCScorer, KNNScorer, LinearScorer, DescriptionScorer, RerankScorer]
)

SCORING_MODULES_MULTILABEL: dict[str, type[ScoringModule]] = _create_modules_dict(
    [MLKnnScorer, LinearScorer, DescriptionScorer],
)

PREDICTION_MODULES_MULTICLASS: dict[str, type[DecisionModule]] = _create_modules_dict(
    [ArgmaxDecision, JinoosDecision, ThresholdDecision, TunableDecision],
)

PREDICTION_MODULES_MULTILABEL: dict[str, type[DecisionModule]] = _create_modules_dict(
    [AdaptiveDecision, ThresholdDecision, TunableDecision],
)

__all__ = []  # type: ignore[var-annotated]
