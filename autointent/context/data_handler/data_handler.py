import logging
from typing import Any, TypedDict

from transformers import set_seed

from autointent.custom_types import LabelType

from .dataset import Dataset, Split
from .multilabel_generation import generate_multilabel_version
from .sampling import sample_from_regex
from .stratification import split
from .tags import collect_tags

logger = logging.getLogger(__name__)


class RegexPatterns(TypedDict):
    id: int
    regexp_full_match: list[str]
    regexp_partial_match: list[str]


class DataAugmenter:
    def __init__(
        self, multilabel_generation_config: str | None = None, regex_sampling: int = 0, random_seed: int = 0
    ) -> None:
        self.multilabel_generation_config = multilabel_generation_config
        self.regex_sampling = regex_sampling
        self.random_seed = random_seed

    def __call__(self, dataset: Dataset) -> Dataset:
        if self.regex_sampling > 0:
            logger.debug(
                "sampling %s utterances from regular expressions for each intent class...", self.regex_sampling
            )
            dataset = sample_from_regex(dataset=dataset, n_shots=self.regex_sampling)

        if self.multilabel_generation_config is not None and self.multilabel_generation_config != "":
            logger.debug("generating multilabel utterances from multiclass ones...")
            dataset = generate_multilabel_version(
                dataset=dataset,
                config_string=self.multilabel_generation_config,
                random_seed=self.random_seed,
            )

        return dataset


class DataHandler:
    def __init__(
        self,
        dataset: Dataset,
        force_multilabel: bool = False,
        random_seed: int = 0,
        augmenter: DataAugmenter | None = None,
    ) -> None:
        set_seed(random_seed)

        self.dataset = dataset
        if force_multilabel:
            self.dataset = self.dataset.to_multilabel()
        if dataset.multilabel:
            self.dataset = self.dataset.encode_labels()

        self.label_descriptions: list[str | None] = [
            intent.description
            for intent in sorted(dataset.intents, key=lambda x: x.id)
        ]

        if augmenter is not None:
            self.dataset = augmenter(self.dataset) # TODO

        self.tags = collect_tags(self.dataset) # TODO (class Tag)

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

        self._logger = logger

    @property
    def multilabel(self) -> bool:
        return self.dataset.multilabel

    @property
    def n_classes(self) -> int:
        return self.dataset.n_classes

    @property
    def train_utterances(self) -> list[str]:
        return self.dataset[Split.TRAIN][self.dataset.UTTERANCE_COLUMN]

    @property
    def train_labels(self) -> list[LabelType]:
        return self.dataset[Split.TRAIN][self.dataset.LABEL_COLUMN]

    @property
    def test_utterances(self) -> list[str]:
        return self.dataset[Split.TEST][self.dataset.UTTERANCE_COLUMN]

    @property
    def test_labels(self) -> list[LabelType]:
        return self.dataset[Split.TEST][self.dataset.LABEL_COLUMN]

    @property
    def oos_utterances(self) -> list[str]:
        if self.has_oos_samples():
            return self.dataset[Split.OOS][self.dataset.UTTERANCE_COLUMN]
        return []

    def has_oos_samples(self) -> bool:
        return Split.OOS in self.dataset

    def dump(self) -> dict[str, list[dict[str, Any]]]:
        return self.dataset.dump()
