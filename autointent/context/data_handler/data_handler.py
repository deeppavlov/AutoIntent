import logging
from typing import Any

from transformers import set_seed

from autointent.custom_types import TASK_MODES

from .multilabel_generation import convert_to_multilabel_format, generate_multilabel_version
from .sampling import sample_from_regex
from .scheme import IntentRecord, UtteranceRecord
from .stratification import split_sample_utterances
from .tags import collect_tags


class DataHandler:
    def __init__(
        self,
        multiclass_intent_records: list[dict[str, Any]],
        multilabel_utterance_records: list[dict[str, Any]],
        test_utterance_records: list[dict[str, Any]],
        mode: TASK_MODES,
        multilabel_generation_config: str | None = None,
        regex_sampling: int = 0,
        seed: int = 0,
    ) -> None:
        logger = logging.getLogger(__name__)
        set_seed(seed)

        # TODO do somthing with this else if
        if not multiclass_intent_records and not multilabel_utterance_records:
            msg = "No data provided, both `multiclass_intent_records` and `multilabel_utterance_records` are empty"
            logger.error(msg)
            raise ValueError(msg)

        if regex_sampling > 0:
            logger.debug("sampling %s utterances from regular expressions for each intent class...", regex_sampling)
            multiclass_intent_records = sample_from_regex(multiclass_intent_records, n_shots=regex_sampling)

        if multilabel_generation_config is not None:
            logger.debug("generating multilabel utterances from multiclass ones...")
            new_utterances = generate_multilabel_version(multiclass_intent_records, multilabel_generation_config, seed)
            multilabel_utterance_records.extend(new_utterances)
            logger.debug("collecting tags from multiclass intent_records if present...")
            self.tags = collect_tags(multiclass_intent_records)

        if mode == "multiclass":
            data = multiclass_intent_records
            self.tags = []

        elif mode == "multilabel":
            data = multilabel_utterance_records
            self.tags = []  # TODO add tags supporting for a pure multilabel case?

        elif mode == "multiclass_as_multilabel":
            if not hasattr(self, "tags"):
                logger.debug("collecting tags from multiclass intent_records if present...")
                self.tags = collect_tags(multiclass_intent_records)

            logger.debug("formatting multiclass labels to multilabel...")
            old_utterances = convert_to_multilabel_format(multiclass_intent_records)
            multilabel_utterance_records.extend(old_utterances)
            data = multilabel_utterance_records

        else:
            msg = f"unexpected classification mode value: {mode}"
            logger.error(msg)
            raise ValueError(msg)

        self.multilabel = mode != "multiclass"

        logger.info("defining train and test splits...")
        (
            self.n_classes,
            self.oos_utterances,
            self.utterances_train,
            self.utterances_test,
            self.labels_train,
            self.labels_test,
        ) = split_sample_utterances(data, test_utterance_records, self.multilabel, seed)

        if mode != "multilabel":
            logger.debug("collection regexp patterns from multiclass intent records")
            self.regexp_patterns = [
                IntentRecord(
                    intent_id=intent["intent_id"],
                    regexp_full_match=intent["regexp_full_match"],
                    regexp_partial_match=intent["regexp_partial_match"],
                )
                for intent in multiclass_intent_records
            ]

        self._logger = logger

    def has_oos_samples(self) -> bool:
        return len(self.oos_utterances) > 0

    def dump(
        self,
    ) -> tuple[list[dict[str, Any] | UtteranceRecord], list[UtteranceRecord]]:
        self._logger.debug("dumping train, test and oos data...")
        train_data = _dump_train(self.utterances_train, self.labels_train, self.n_classes, self.multilabel)
        test_data = _dump_test(self.utterances_test, self.labels_test, self.n_classes, self.multilabel)
        oos_data = _dump_oos(self.oos_utterances)
        test_data = test_data + oos_data
        return train_data, test_data


def _dump_train(
    utterances: list[str],
    labels: list[list[int]] | list[int],
    n_classes: int,
    multilabel: bool,
) -> list[dict[str, Any] | UtteranceRecord]:
    if multilabel and isinstance(labels[0], list):
        res = []
        for ut, labs in zip(utterances, labels, strict=False):
            labs_converted = [i for i in range(n_classes) if labs[i]]
            res.append(UtteranceRecord(utterance=ut, labels=labs_converted))
    elif not multilabel and isinstance(labels[0], int):
        # TODO check if rec is used
        res = [{"intent_id": i} for i in range(n_classes)]
        for ut, lab in zip(utterances, labels, strict=False):
            rec = res[lab]
            rec["sample_utterances"] = [*rec.get("sample_utterances", []), ut]
    else:
        message = "unexpected labels format"
        raise ValueError(message)
    return res


def _dump_test(
    utterances: list[str],
    labels: list[list[int]] | list[int],
    n_classes: int,
    multilabel: bool,
) -> list[UtteranceRecord]:
    res = []
    for ut, labs in zip(utterances, labels, strict=True):
        labs_converted = (
            [i for i in range(n_classes) if labs[i]] if multilabel and isinstance(labels[0], list) else [labs]
        )
        res.append(UtteranceRecord(utterance=ut, labels=labs_converted))
    return res


def _dump_oos(utterances: list[str]) -> list[UtteranceRecord]:
    return [UtteranceRecord(utterance=ut, labels=[]) for ut in utterances]
