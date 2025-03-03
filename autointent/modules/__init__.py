"""Heart of the library with different intent classification methods implemented."""

from typing import TypeVar

from .base import BaseDecision, BaseEmbedding, BaseModule, BaseRegex, BaseScorer
from .decision import (
    AdaptiveDecision,
    ArgmaxDecision,
    JinoosDecision,
    ThresholdDecision,
    TunableDecision,
)
from .embedding import LogregAimedEmbedding, RetrievalAimedEmbedding
from .regex import SimpleRegex
from .scoring import DescriptionScorer, DNNCScorer, KNNScorer, LinearScorer, MLKnnScorer, RerankScorer, SklearnScorer

T = TypeVar("T", bound=BaseModule)


def _create_modules_dict(modules: list[type[T]]) -> dict[str, type[T]]:
    return {module.name: module for module in modules}


REGEX_MODULES: dict[str, type[BaseRegex]] = _create_modules_dict([SimpleRegex])

EMBEDDING_MODULES: dict[str, type[BaseEmbedding]] = _create_modules_dict(
    [RetrievalAimedEmbedding, LogregAimedEmbedding]
)

SCORING_MODULES: dict[str, type[BaseScorer]] = _create_modules_dict(
    [
        DNNCScorer,
        KNNScorer,
        LinearScorer,
        DescriptionScorer,
        RerankScorer,
        SklearnScorer,
        MLKnnScorer,
    ]
)

DECISION_MODULES: dict[str, type[BaseDecision]] = _create_modules_dict(
    [ArgmaxDecision, JinoosDecision, ThresholdDecision, TunableDecision, AdaptiveDecision],
)


__all__ = []  # type: ignore[var-annotated]
