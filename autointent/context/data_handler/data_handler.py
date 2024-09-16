from typing import Literal

from .multilabel_generation import convert_to_multilabel_format, generate_multilabel_version
from .sampling import sample_from_regex
from .stratification import split_sample_utterances
from .tags import collect_tags

import logging
logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(
        self,
        multiclass_intent_records: list[dict],
        multilabel_utterance_records: list[dict],
        test_utterance_records: list[dict],
        mode: Literal["multiclass", "multilabel", "multiclass_as_multilabel"],
        multilabel_generation_config: str = "",
        regex_sampling: int = 0,
        seed: int = 0,
    ):
        if not multiclass_intent_records and not multilabel_utterance_records:
            raise ValueError("you must provide some data")

        if regex_sampling > 0:
            sample_from_regex(multiclass_intent_records, n_shots=regex_sampling)

        if multilabel_generation_config != "":
            new_utterances = generate_multilabel_version(multiclass_intent_records, multilabel_generation_config, seed)
            multilabel_utterance_records.extend(new_utterances)
            self.tags = collect_tags(multiclass_intent_records)

        if mode == "multiclass":
            data = multiclass_intent_records
            self.tags = []

        elif mode == "multilabel":
            data = multilabel_utterance_records
            self.tags = []  # TODO add tags supporting for a pure multilabel case?

        elif mode == "multiclass_as_multilabel":
            if not hasattr(self, "tags"):
                self.tags = collect_tags(multiclass_intent_records)
            old_utterances = convert_to_multilabel_format(multiclass_intent_records)
            multilabel_utterance_records.extend(old_utterances)
            data = multilabel_utterance_records

        else:
            raise ValueError(f"unexpected mode value: {mode}")

        self.multilabel = mode != "multiclass"

        (
            self.n_classes,
            self.oos_utterances,
            self.utterances_train,
            self.utterances_test,
            self.labels_train,
            self.labels_test,
        ) = split_sample_utterances(data, test_utterance_records, self.multilabel, seed)

        if mode != "multilabel":
            self.regexp_patterns = [
                dict(
                    intent_id=intent["intent_id"],
                    regexp_full_match=intent["regexp_full_match"],
                    regexp_partial_match=intent["regexp_partial_match"],
                )
                for intent in multiclass_intent_records
            ]

    def has_oos_samples(self):
        return len(self.oos_utterances) > 0

    def dump(self):
        train_data = _dump_train(self.utterances_train, self.labels_train, self.n_classes, self.multilabel)
        test_data = _dump_test(self.utterances_test, self.labels_test, self.n_classes, self.multilabel)
        oos_data = _dump_oos(self.oos_utterances)
        test_data = test_data + oos_data
        return train_data, test_data

    def print_fields(self):
        logger.info("DataHandler fields:")
        logger.info(f"Multilabel: {self.multilabel}")
        logger.info(f"Number of classes: {self.n_classes}")
        logger.info(f"Number of training utterances: {len(self.utterances_train)}")
        logger.info(f"Number of test utterances: {len(self.utterances_test)}")
        logger.info(f"Number of OOS utterances: {len(self.oos_utterances)}")
        logger.info(f"Number of tags: {len(self.tags)}")
        if hasattr(self, 'regexp_patterns'):
            logger.info(f"Number of regexp patterns: {len(self.regexp_patterns)}")

        if self.utterances_train:
            logger.info(f"Sample training utterance: {self.utterances_train[0]}")
            logger.info(f"Sample training label: {self.labels_train[0]}")
        if self.utterances_test:
            logger.info(f"Sample test utterance: {self.utterances_test[0]}")
            logger.info(f"Sample test label: {self.labels_test[0]}")
        if self.oos_utterances:
            logger.info(f"Sample OOS utterance: {self.oos_utterances[0]}")
        if self.tags:
            logger.info(f"Sample tag: {self.tags[0]}")
        if hasattr(self, 'regexp_patterns') and self.regexp_patterns:
            logger.info(f"Sample regexp pattern: {self.regexp_patterns[0]}")


def _dump_train(utterances, labels, n_classes, multilabel):
    if not multilabel:
        res = [dict(intent_id=i) for i in range(n_classes)]
        for ut, lab in zip(utterances, labels):
            rec = res[lab]
            sample_utterances = rec.get("sample_utterances", []) + [ut]
            rec["sample_utterances"] = sample_utterances
    else:
        res = []
        for ut, labs in zip(utterances, labels):
            labs = [i for i in range(n_classes) if labs[i]]
            res.append(dict(utterance=ut, labels=labs))
    return res
    

def _dump_test(utterances, labels, n_classes, multilabel):
    res = []
    for ut, labs in zip(utterances, labels):
        if multilabel:
            labs = [i for i in range(n_classes) if labs[i]]
        else:
            labs = [labs]
        res.append(dict(utterance=ut, labels=labs))
    return res


def _dump_oos(utterances):
    return [dict(utterance=ut, labels=[]) for ut in utterances]
