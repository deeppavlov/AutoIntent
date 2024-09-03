import os

from .stratification import split_sample_utterances
from .sampling import sample_from_regex


class DataHandler:
    def __init__(self, intent_records: os.PathLike, multilabel: bool, regex_sampling: int = 0, seed: int = 0):
        if regex_sampling > 0:
            sample_from_regex(intent_records, n_shots=regex_sampling)

        (
            self.n_classes,
            self.oos_utterances,
            self.utterances_train,
            self.utterances_test,
            self.labels_train,
            self.labels_test,
        ) = split_sample_utterances(intent_records, multilabel, seed)

        if not multilabel:
            self.regexp_patterns = [
                dict(
                    intent_id=intent["intent_id"],
                    regexp_full_match=intent["regexp_full_match"],
                    regexp_partial_match=intent["regexp_partial_match"],
                )
                for intent in intent_records
            ]
