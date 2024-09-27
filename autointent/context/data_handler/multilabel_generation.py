import json
import random
from itertools import combinations

import xeger

from .stratification import get_sample_utterances


def sample_unique_tuples(k, n, m):
    all_combinations = list(combinations(range(n), k))
    random.shuffle(all_combinations)
    return all_combinations[:m]


def sample_utterance_from_regexp(intent_record: dict, x: xeger.Xeger):
    n_templates = len(intent_record["regexp_full_match"])
    i_template = random.randint(0, n_templates - 1)
    res = x.xeger(intent_record["regexp_full_match"][i_template])
    return res.strip()


def sample_multilabel_utterances(intent_records: list[dict], n_samples, n_labels, seed=0):
    # TODO improve versatility
    random.seed(seed)
    x = xeger.Xeger()
    x.seed(seed)
    n_given_intents = len(intent_records)
    res = []
    for t in sample_unique_tuples(n_labels, n_given_intents, n_samples):
        sampled_utterances = [sample_utterance_from_regexp(intent_records[i], x) for i in t]
        utterance = ". ".join(sampled_utterances)
        res.append({"utterance": utterance, "labels": t})
    return res


def generate_multilabel_version(intent_records, config_string, seed):
    config = json.loads(config_string)
    res = []
    for i in range(len(config)):
        new_records = sample_multilabel_utterances(intent_records, n_samples=int(config[i]), n_labels=i + 1, seed=seed)
        res.extend(new_records)
    return res


def convert_to_multilabel_format(intent_records):
    utterances, labels = get_sample_utterances(intent_records)
    return [{"utterance": ut, "labels": [lab] if lab != -1 else []} for ut, lab in zip(utterances, labels, strict=False)]
