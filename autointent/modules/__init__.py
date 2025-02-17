"""Heart of the library with different intent classification methods implemented."""

from typing import TypeVar

from .abc import BaseDecision, BaseEmbedding, BaseModule, BaseRegex, BaseScorer
from .decision import (
    AdaptiveDecision,
    ArgmaxDecision,
    JinoosDecision,
    ThresholdDecision,
    TunableDecision,
)
from .embedding import LogregAimedEmbedding, RetrievalAimedEmbedding
from .regex import Regex
from .scoring import DescriptionScorer, DNNCScorer, KNNScorer, LinearScorer, MLKnnScorer, RerankScorer, SklearnScorer

T = TypeVar("T", bound=BaseModule)


def _create_modules_dict(modules: list[type[T]]) -> dict[str, type[T]]:
    return {module.name: module for module in modules}


REGEX_MODULES: dict[str, type[BaseRegex]] = _create_modules_dict([Regex])

EMBEDDING_MODULES_MULTICLASS: dict[str, type[BaseEmbedding]] = _create_modules_dict(
    [RetrievalAimedEmbedding, LogregAimedEmbedding]
)

EMBEDDING_MODULES_MULTILABEL: dict[str, type[BaseEmbedding]] = EMBEDDING_MODULES_MULTICLASS

SCORING_MODULES_MULTICLASS: dict[str, type[BaseScorer]] = _create_modules_dict(
    [
        DNNCScorer,
        KNNScorer,
        LinearScorer,
        DescriptionScorer,
        RerankScorer,
        SklearnScorer,
    ]
)

SCORING_MODULES_MULTILABEL: dict[str, type[BaseScorer]] = _create_modules_dict(
    [
        MLKnnScorer,
        LinearScorer,
        DescriptionScorer,
        SklearnScorer,
    ],
)

DECISION_MODULES_MULTICLASS: dict[str, type[BaseDecision]] = _create_modules_dict(
    [ArgmaxDecision, JinoosDecision, ThresholdDecision, TunableDecision],
)

DECISION_MODULES_MULTILABEL: dict[str, type[BaseDecision]] = _create_modules_dict(
    [AdaptiveDecision, ThresholdDecision, TunableDecision],
)

__all__ = []  # type: ignore[var-annotated]
