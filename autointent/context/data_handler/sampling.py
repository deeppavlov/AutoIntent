"""Module for generating samples from regex templates.

This module provides utilities for generating synthetic datasets using regex
templates, distributing samples among bins, and augmenting datasets with
generated utterances.
"""

import random

import sre_yield

from .schemas import Dataset, Utterance


def distribute_shots(n: int, k: int) -> list[int]:
    """
    Randomly distribute `k` samples among `n` bins.

    This function divides a given number of samples (`k`) into `n` bins
    (e.g., patterns or classes) in a randomized manner.

    :param n: Number of bins to distribute samples into.
    :param k: Total number of samples to distribute.
    :return: A list of integers where each value represents the number of
             samples assigned to the corresponding bin.
    """
    samples_per_bin = [0] * n
    for _ in range(k):
        i_bin = random.randint(0, n - 1)
        samples_per_bin[i_bin] += 1
    return samples_per_bin


def generate_from_template(template: str, n: int) -> list[str]:
    """
    Generate samples from a regex template.

    Uses `sre_yield` to generate strings matching a given regex template. The
    function returns up to `n` matching samples, or fewer if the total
    possible matches is smaller than `n`.

    :param template: A regex template string for generating samples.
    :param n: Maximum number of samples to generate.
    :return: A list of strings generated from the regex template.
    """
    return list(sre_yield.AllStrings(template))[:n]


def generate_from_templates(patterns: list[str], n_shots: int) -> list[str]:
    """
    Generate samples from multiple regex templates.

    Distributes `n_shots` among the provided regex templates and generates
    samples accordingly. The samples are aggregated from all patterns.

    :param patterns: A list of regex templates to generate samples from.
    :param n_shots: Total number of samples to generate across all templates.
    :return: A list of strings generated from the provided templates.
    """
    shots_per_pattern = distribute_shots(len(patterns), n_shots)
    res = []
    for pattern, n in zip(patterns, shots_per_pattern, strict=False):
        new_samples = generate_from_template(pattern, n)
        res.extend(new_samples)
    return res


def sample_from_regex(
    dataset: Dataset,
    n_shots: int,
    random_seed: int = 0,
) -> Dataset:
    """
    Augment a dataset with samples generated from regex templates.

    Generates synthetic utterances for each intent in the dataset by sampling
    from the provided regular expression templates. The generated utterances
    are added to the dataset with their corresponding labels.

    :param dataset: A `Dataset` object containing intents and regex templates.
    :param n_shots: Total number of samples to generate for each intent.
    :param random_seed: Random seed for reproducibility.
    :return: The augmented `Dataset` object containing the newly generated utterances.
    """
    random.seed(random_seed)

    for intent in dataset.intents:
        generated_texts = generate_from_templates(intent.regexp_full_match, n_shots)
        for generated_text in generated_texts:
            utterance = Utterance(text=generated_text, label=intent.id)
            dataset.utterances.append(utterance)

    return dataset
