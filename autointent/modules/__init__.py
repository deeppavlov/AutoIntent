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
from .scoring import (
    BERTLoRAScorer,
    BertScorer,
    BiEncoderDescriptionScorer,
    CatBoostScorer,
    CNNScorer,
    CrossEncoderDescriptionScorer,
    DNNCScorer,
    KNNScorer,
    LinearScorer,
    LLMDescriptionScorer,
    MLKnnScorer,
    PTuningScorer,
    RerankScorer,
    RNNScorer,
    SklearnScorer,
)

T = TypeVar("T", bound=BaseModule)


def _create_modules_dict(modules: list[type[T]]) -> dict[str, type[T]]:
    return {module.name: module for module in modules}


REGEX_MODULES: dict[str, type[BaseRegex]] = _create_modules_dict([SimpleRegex])

EMBEDDING_MODULES: dict[str, type[BaseEmbedding]] = _create_modules_dict(
    [RetrievalAimedEmbedding, LogregAimedEmbedding]
)

SCORING_MODULES: dict[str, type[BaseScorer]] = _create_modules_dict(
    [
        CatBoostScorer,
        DNNCScorer,
        KNNScorer,
        LinearScorer,
        BiEncoderDescriptionScorer,
        CrossEncoderDescriptionScorer,
        LLMDescriptionScorer,
        RerankScorer,
        SklearnScorer,
        MLKnnScorer,
        BertScorer,
        CNNScorer,
        BERTLoRAScorer,
        PTuningScorer,
        RNNScorer,
    ]
)

DECISION_MODULES: dict[str, type[BaseDecision]] = _create_modules_dict(
    [ArgmaxDecision, JinoosDecision, ThresholdDecision, TunableDecision, AdaptiveDecision],
)
