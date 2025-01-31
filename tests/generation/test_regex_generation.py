import importlib.resources as ires

import pytest

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation.regex_generation import sample_from_regex


@pytest.fixture
def in_dataset():
    data_path = ires.files("tests.assets.data").joinpath("dream_subset.json")
    return Dataset.from_json(data_path)


def test_generation_basic(in_dataset):
    result = sample_from_regex(in_dataset, n_shots=3)

    assert len(result[Split.TRAIN]) == 3
    assert len(result[Split.VALIDATION]) == 3
    assert len(result[Split.TEST]) == 3


def test_generation_all_samples(in_dataset):
    result = sample_from_regex(in_dataset, n_shots=1000)

    assert len(result[Split.TRAIN]) == 1273
    assert len(result[Split.VALIDATION]) == 424
    assert len(result[Split.TEST]) == 425


def test_generation_deterministic(in_dataset):
    result1 = sample_from_regex(in_dataset, n_shots=3, random_seed=42)
    result2 = sample_from_regex(in_dataset, n_shots=3, random_seed=42)

    assert len(result1[Split.TRAIN]) != 0
    assert result1[Split.TRAIN][Dataset.utterance_feature] == result2[Split.TRAIN][Dataset.utterance_feature]

    assert len(result1[Split.VALIDATION]) != 0
    assert result1[Split.VALIDATION][Dataset.utterance_feature] == result2[Split.VALIDATION][Dataset.utterance_feature]

    assert len(result1[Split.TEST]) != 0
    assert result1[Split.TEST][Dataset.utterance_feature] == result2[Split.TEST][Dataset.utterance_feature]


def test_generation_deterministic_different_seed(in_dataset):
    result1 = sample_from_regex(in_dataset, n_shots=3, random_seed=42)
    result2 = sample_from_regex(in_dataset, n_shots=3, random_seed=40)

    assert len(result1[Split.TRAIN]) != 0
    assert result1[Split.TRAIN][Dataset.utterance_feature] != result2[Split.TRAIN][Dataset.utterance_feature]

    assert len(result1[Split.VALIDATION]) != 0
    assert result1[Split.VALIDATION][Dataset.utterance_feature] != result2[Split.VALIDATION][Dataset.utterance_feature]

    assert len(result1[Split.TEST]) != 0
    assert result1[Split.TEST][Dataset.utterance_feature] != result2[Split.TEST][Dataset.utterance_feature]
