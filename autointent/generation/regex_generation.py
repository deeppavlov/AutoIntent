"""Dataset generation from regexp."""

from random import Random

from sklearn.model_selection import train_test_split

from autointent import Dataset
from autointent.custom_types import Split

# until the changes are accepted, or we make our own release
from autointent.generation import exrex


def _sample_intent_regexp(
    regexps: list[str], n_shots: int, n_rep_limit: int, label: int | list[int], rng: Random
) -> list[dict]:  # type: ignore[type-arg]
    generators = [exrex.generate(regex, limit=n_rep_limit) for regex in regexps]  # type: ignore[attr-defined]
    results = []

    while n_shots and len(generators) > 0:
        idx = rng.randint(0, len(generators) - 1)
        try:
            results.append({Dataset.utterance_feature: next(generators[idx]), Dataset.label_feature: label})
            n_shots -= 1
        except StopIteration:
            generators[idx] = generators[-1]
            generators.pop()

    return results


def sample_from_regex(
    in_dataset: Dataset,
    n_shots: int,
    n_rep_limit: int = 20,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_seed: int | None = None,
) -> Dataset:
    """
    Generate utterances from dataset with regular expressions.

    :param in_dataset: The dataset containing intents with regular exressions.
    :param n_shots: The maximum number of samples to produce for every intent.
    :param n_rep_limit: To limit the number of possible repetitions in a regular expression.
    :param val_size: The proportion to be allocated for the validation part.
    :param test_size: The proportion to be allocated for the test part.
    :param random_seed: To make your sampling deterministic.

    :returns: The dataset with sampled utterances.
    """
    rng = Random(random_seed)
    intents = in_dataset.intents

    splits: dict[str, list] = {  # type: ignore[type-arg]
        Split.TRAIN: [],
        Split.VALIDATION: [],
        Split.TEST: [],
    }

    for intent in intents:
        utterances = _sample_intent_regexp(intent.regexp_full_match, n_shots, n_rep_limit, intent.id, rng)

        x_train, x_remaining = train_test_split(utterances, test_size=val_size + test_size, random_state=random_seed)
        splits[Split.TRAIN].extend(x_train)

        x_val, x_test = train_test_split(
            x_remaining, test_size=test_size / (test_size + val_size), random_state=random_seed
        )

        splits[Split.VALIDATION].extend(x_val)
        splits[Split.TEST].extend(x_test)

    splits["intents"] = intents

    return Dataset.from_dict(splits)
