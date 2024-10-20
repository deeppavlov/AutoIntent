import logging
from collections.abc import Sequence
from typing import Any, TypedDict

from transformers import set_seed

from autointent.custom_types import LABEL_TYPE

from .multilabel_generation import generate_multilabel_version
from .sampling import sample_from_regex
from .schemas import Dataset, DatasetType
from .scheme import UtteranceRecord
from .stratification import split_sample_utterances
from .tags import collect_tags


class RegexPatterns(TypedDict):
    id: int
    regexp_full_match: list[str]
    regexp_partial_match: list[str]


class DataHandler:
    def __init__(
        self,
        dataset: Dataset,
        test_dataset: Dataset | None = None,
        multilabel_generation_config: str = "",
        regex_sampling: int = 0,
        force_multilabel: bool = False,
        random_seed: int = 0,
    ) -> None:
        logger = logging.getLogger(__name__)
        set_seed(random_seed)

        if force_multilabel:
            dataset = dataset.to_multilabel()

        self.multilabel = dataset.type == DatasetType.multilabel

        if regex_sampling > 0:
            logger.debug("sampling %s utterances from regular expressions for each intent class...", regex_sampling)
            dataset = sample_from_regex(dataset=dataset, n_shots=regex_sampling)

        if multilabel_generation_config is not None and multilabel_generation_config != "":
            logger.debug("generating multilabel utterances from multiclass ones...")
            dataset = generate_multilabel_version(
                dataset=dataset,
                config_string=multilabel_generation_config,
                random_seed=random_seed,
            )

        logger.debug("collecting tags from multiclass intent_records if present...")
        self.tags = collect_tags(dataset)

        logger.info("defining train and test splits...")
        (
            self.n_classes,
            self.oos_utterances,
            self.utterances_train,
            self.utterances_test,
            self.labels_train,
            self.labels_test,
        ) = split_sample_utterances(
            dataset=dataset,
            test_dataset=test_dataset,
            random_seed=random_seed,
        )

        logger.debug("collection regexp patterns from multiclass intent records")
        self.regexp_patterns = [
            RegexPatterns(
                id=intent.id,
                regexp_full_match=intent.regexp_full_match,
                regexp_partial_match=intent.regexp_partial_match,
            )
            for intent in dataset.intents
        ]

        self._logger = logger

    def has_oos_samples(self) -> bool:
        return len(self.oos_utterances) > 0

    def dump(
        self,
    ) -> tuple[list[dict[str, Any]], list[UtteranceRecord]]:
        self._logger.debug("dumping train, test and oos data...")
        train_data = _dump_train(self.utterances_train, self.labels_train, self.n_classes, self.multilabel)
        test_data = _dump_test(self.utterances_test, self.labels_test, self.n_classes, self.multilabel)
        oos_data = _dump_oos(self.oos_utterances)
        test_data = test_data + oos_data
        return train_data, test_data  # type: ignore[return-value]


def _dump_train(
    utterances: list[str],
    labels: list[LABEL_TYPE],
    n_classes: int,
    multilabel: bool,
) -> Sequence[dict[str, Any]]:
    if multilabel and isinstance(labels[0], list):
        res = []
        for ut, labs in zip(utterances, labels, strict=True):
            labs_converted = [i for i in range(n_classes) if labs[i]]  # type: ignore[index]
            res.append({"utterance": ut, "labels": labs_converted})
    elif not multilabel and isinstance(labels[0], int):
        # TODO check if rec is used
        res = [{"intent_id": i} for i in range(n_classes)]  # type: ignore[dict-item]
        for ut, lab in zip(utterances, labels, strict=False):
            rec = res[lab]  # type: ignore[index]
            rec["sample_utterances"] = [*rec.get("sample_utterances", []), ut]
    else:
        message = "unexpected labels format"
        raise ValueError(message)
    return res


def _dump_test(
    utterances: list[str],
    labels: list[LABEL_TYPE],
    n_classes: int,
    multilabel: bool,
) -> list[UtteranceRecord]:
    res = []
    for ut, labs in zip(utterances, labels, strict=True):
        labs_converted = (
            [i for i in range(n_classes) if labs[i]] if multilabel and isinstance(labels[0], list) else [labs]  # type: ignore[index,list-item]
        )
        res.append(UtteranceRecord(utterance=ut, labels=labs_converted))
    return res


def _dump_oos(utterances: list[str]) -> list[UtteranceRecord]:
    return [UtteranceRecord(utterance=ut, labels=[]) for ut in utterances]
