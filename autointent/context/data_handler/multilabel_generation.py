import json
import random
from itertools import combinations
from pathlib import Path

import xeger

from .schemas import Dataset, Intent, Utterance


def sample_unique_tuples(k: int, n: int, m: int) -> list[tuple[int, ...]]:
    all_combinations = list(combinations(range(n), k))
    random.shuffle(all_combinations)
    return all_combinations[:m]


def sample_utterance_from_regexp(intent: Intent, x: xeger.Xeger) -> str:
    n_templates = len(intent.regexp_full_match)
    i_template = random.randint(0, n_templates - 1)
    res = x.xeger(intent.regexp_full_match[i_template])
    return res.strip()


def sample_multilabel_utterances(
    dataset: Dataset, n_samples: int, n_labels: int, random_seed: int,
) -> list[Utterance]:
    # TODO improve versatility
    random.seed(random_seed)
    x = xeger.Xeger()
    x.seed(random_seed)
    n_classes = len(dataset.intents)

    sampled_utterances = []
    for t in sample_unique_tuples(n_labels, n_classes, n_samples):
        sampled_texts = [sample_utterance_from_regexp(dataset.intents[i], x) for i in t]
        text = ". ".join(sampled_texts)
        sampled_utterances.append(Utterance(text=text, label=t))
    return sampled_utterances


def generate_multilabel_version(
    dataset: Dataset, config_string: str, random_seed: int,
) -> Dataset:
    config_path = Path(config_string)
    if not config_path.exists():
        msg = f"Config file {config_path} not found"
        raise FileNotFoundError(msg)
    config = json.loads(config_string)

    sampled_utterances = []
    for i in range(len(config)):
        sampled_utterances.extend(
            sample_multilabel_utterances(
                dataset=dataset,
                n_samples=int(config[i]),
                n_labels=i + 1,
                random_seed=random_seed,
            ),
        )

    dataset.utterances.extend(sampled_utterances)

    return dataset
