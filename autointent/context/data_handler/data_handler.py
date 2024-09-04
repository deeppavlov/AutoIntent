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
        ) = split_sample_utterances(data, self.multilabel, seed)

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
