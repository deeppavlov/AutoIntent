import logging
from typing import Any, TypedDict

from transformers import set_seed

from autointent.custom_types import LabelType

from .dataset import Dataset, Split
from .stratification import split

logger = logging.getLogger(__name__)


class RegexPatterns(TypedDict):
    id: int
    regexp_full_match: list[str]
    regexp_partial_match: list[str]


class DataHandler:
    def __init__(
        self,
        dataset: Dataset,
        force_multilabel: bool = False,
        random_seed: int = 0,
    ) -> None:
        set_seed(random_seed)

        self.dataset = dataset
        if force_multilabel:
            self.dataset = self.dataset.to_multilabel()
        if dataset.multilabel:
            self.dataset = self.dataset.encode_labels()

        if Split.TEST not in self.dataset:
            logger.info("Spltting dataset into train and test splits")
            self.dataset = split(self.dataset, random_seed=random_seed)

        self.regexp_patterns = [
            RegexPatterns(
                id=intent.id,
                regexp_full_match=intent.regexp_full_match,
                regexp_partial_match=intent.regexp_partial_match,
            )
            for intent in dataset.intents
        ]

        self.intent_descriptions = [intent.id for intent in dataset.intents]
        self.tags = self.dataset.get_tags()

        self._logger = logger

    @property
    def multilabel(self) -> bool:
        return self.dataset.multilabel

    @property
    def n_classes(self) -> int:
        return self.dataset.n_classes

    @property
    def train_utterances(self) -> list[str]:
        return self.dataset[Split.TRAIN][self.dataset.utterance_feature]

    @property
    def train_labels(self) -> list[LabelType]:
        return self.dataset[Split.TRAIN][self.dataset.label_feature]

    @property
    def test_utterances(self) -> list[str]:
        return self.dataset[Split.TEST][self.dataset.utterance_feature]

    @property
    def test_labels(self) -> list[LabelType]:
        return self.dataset[Split.TEST][self.dataset.label_feature]

    @property
    def oos_utterances(self) -> list[str]:
        if self.has_oos_samples():
            return self.dataset[Split.OOS][self.dataset.utterance_feature]
        return []

    def has_oos_samples(self) -> bool:
        return Split.OOS in self.dataset

    def dump(self) -> dict[str, list[dict[str, Any]]]:
        return self.dataset.dump()
