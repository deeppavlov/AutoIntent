"""Module for generating multilabel datasets.

This module provides functions for sampling unique tuples, generating utterances
from regular expressions, and creating multilabel datasets for use in natural
language processing tasks.
"""

import json
import random
from itertools import combinations

import xeger

from ._schemas import Dataset, Intent, Utterance


def sample_unique_tuples(k: int, n: int, m: int) -> list[tuple[int, ...]]:
    """
    Sample `m` unique tuples of size `k` from a range of `n` elements.

    This function generates all possible combinations of `k` elements from a
    range of `n`, shuffles them, and returns the first `m` unique tuples.

    :param k: Number of elements in each tuple.
    :param n: Total number of elements to choose from.
    :param m: Number of unique tuples to sample.
    :return: A list of `m` unique tuples, each containing `k` elements.
    """
    all_combinations = list(combinations(range(n), k))
    random.shuffle(all_combinations)
    return all_combinations[:m]


def sample_utterance_from_regexp(intent: Intent, x: xeger.Xeger) -> str:
    """
    Generate a sample utterance from a regular expression.

    Randomly selects one of the full-match regular expressions defined in the
    `Intent` object and uses `xeger` to generate a matching string.

    :param intent: An `Intent` object containing regular expressions.
    :param x: An instance of `xeger.Xeger` used for regex-based string generation.
    :return: A string generated from one of the intent's regular expressions.
    """
    n_templates = len(intent.regexp_full_match)
    i_template = random.randint(0, n_templates - 1)
    res = x.xeger(intent.regexp_full_match[i_template])
    return res.strip()  # type: ignore[no-any-return]


def sample_multilabel_utterances(
    dataset: Dataset,
    n_samples: int,
    n_labels: int,
    random_seed: int,
) -> list[Utterance]:
    """
    Sample multilabel utterances from a dataset.

    Combines multiple intents' utterances into single multilabel utterances
    with a specified number of labels. The text for each label is generated
    from the intents' regular expressions.

    :param dataset: A `Dataset` object containing intents and their regular expressions.
    :param n_samples: Number of multilabel utterances to sample.
    :param n_labels: Number of labels to assign to each multilabel utterance.
    :param random_seed: Random seed for reproducibility.
    :return: A list of `Utterance` objects with multilabel annotations.
    """
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
    dataset: Dataset,
    config_string: str,
    random_seed: int,
) -> Dataset:
    """
    Generate a multilabel version of a dataset.

    Creates a multilabel dataset by sampling utterances based on the specified
    configuration. The configuration defines the number of samples for each
    number of labels.

    :param dataset: A `Dataset` object containing intents and their regular expressions.
    :param config_string: A JSON string representing the sampling configuration.
                          Each index specifies the number of samples for a specific
                          number of labels (e.g., index 0 for single-label, 1 for
                          two-label combinations, etc.).
    :param random_seed: Random seed for reproducibility.
    :return: A modified `Dataset` object containing additional multilabel utterances.
    """
    configs = json.loads(config_string)

    sampled_utterances = []
    for i, config in enumerate(configs, start=1):
        sampled_utterances.extend(
            sample_multilabel_utterances(
                dataset=dataset,
                n_samples=int(config),
                n_labels=i,
                random_seed=random_seed,
            ),
        )

    dataset.utterances.extend(sampled_utterances)

    return dataset
