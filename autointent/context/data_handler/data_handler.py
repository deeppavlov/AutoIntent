import logging
from typing import Literal

from .multilabel_generation import convert_to_multilabel_format, generate_multilabel_version
from .sampling import sample_from_regex
from .stratification import split_sample_utterances
from .tags import collect_tags


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
        logger = logging.getLogger(__name__)

        if not multiclass_intent_records and not multilabel_utterance_records:
            msg = "No data provided, both `multiclass_intent_records` and `multilabel_utterance_records` are empty"
            logger.error(msg)
            raise ValueError(msg)

        if regex_sampling > 0:
            logger.debug(f"sampling {regex_sampling} utterances from regular expressions for each intent class...")
            sample_from_regex(multiclass_intent_records, n_shots=regex_sampling)

        if multilabel_generation_config != "":
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
                {
                    "intent_id": intent["intent_id"],
                    "regexp_full_match": intent["regexp_full_match"],
                    "regexp_partial_match": intent["regexp_partial_match"],
                }
                for intent in multiclass_intent_records
            ]

        self._logger = logger

    def has_oos_samples(self):
        return len(self.oos_utterances) > 0

    def dump(self):
        self._logger.debug("dumping train, test and oos data...")
        train_data = _dump_train(self.utterances_train, self.labels_train, self.n_classes, self.multilabel)
        test_data = _dump_test(self.utterances_test, self.labels_test, self.n_classes, self.multilabel)
        oos_data = _dump_oos(self.oos_utterances)
        test_data = test_data + oos_data
        return train_data, test_data


def _dump_train(utterances, labels, n_classes, multilabel):
    if not multilabel:
        res = [{"intent_id": i} for i in range(n_classes)]
        for ut, lab in zip(utterances, labels, strict=False):
            rec = res[lab]
            sample_utterances = [*rec.get("sample_utterances", []), ut]
            rec["sample_utterances"] = sample_utterances
    else:
        res = []
        for ut, labs in zip(utterances, labels, strict=False):
            labs = [i for i in range(n_classes) if labs[i]]
            res.append({"utterance": ut, "labels": labs})
    return res


def _dump_test(utterances, labels, n_classes, multilabel):
    res = []
    for ut, labs in zip(utterances, labels, strict=False):
        labs = [i for i in range(n_classes) if labs[i]] if multilabel else [labs]
        res.append({"utterance": ut, "labels": labs})
    return res


def _dump_oos(utterances):
    return [{"utterance": ut, "labels": []} for ut in utterances]
