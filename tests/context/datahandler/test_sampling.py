import re

import pytest

from autointent.context.data_handler import Dataset
from autointent.context.data_handler.sampling import (
    distribute_shots,
    generate_from_template,
    generate_from_templates,
    sample_from_regex,
)


def test_distribute_shots():
    assert sum(distribute_shots(5, 10)) == 10
    assert len(distribute_shots(5, 10)) == 5
    assert all(x >= 0 for x in distribute_shots(5, 10))

    assert distribute_shots(1, 10) == [10]
    assert distribute_shots(10, 0) == [0] * 10


def test_generate_from_template():
    assert set(generate_from_template("a|b", 10)) == {"a", "b"}
    assert len(generate_from_template("\\d", 10)) == 10
    assert all(re.match("\\d", x) for x in generate_from_template("\\d", 10))

    assert len(generate_from_template("a|b", 3)) == 2


def test_sample_from_regex():
    data = {
        "utterances": [{"text": "hello", "label": 0}, {"text": "goodbye", "label": 1}],
        "intents": [
            {"id": 0, "regexp_full_match": ["hello|hi", "hey there"]},
            {"id": 1, "regexp_full_match": ["goodbye|bye"]},
        ],
    }

    n_shots = 5
    dataset = sample_from_regex(Dataset.model_validate(data), n_shots)

    assert len(dataset.intents) == 2
    for intent in dataset.intents:
        for utterance in dataset.utterances[2:]:
            if intent.id == utterance.label:
                assert any(re.fullmatch(pattern, utterance.text) for pattern in intent.regexp_full_match)


@pytest.mark.parametrize(("n", "k"), [(5, 10), (1, 10), (10, 0), (100, 1000)])
def test_distribute_shots_properties(n: int, k: int):
    result = distribute_shots(n, k)
    assert len(result) == n
    assert sum(result) == k
    assert all(isinstance(x, int) and x >= 0 for x in result)


def test_generate_from_template_limit():
    assert len(generate_from_template("a|b", 100)) == 2


def test_generate_from_templates_distribution():
    patterns = ["a|b", "c|d|e", "f"]
    result = generate_from_templates(patterns, 100)

    assert len(result) == 6
    assert all(x in "abcdef" for x in result)


def test_sample_from_regex_consistency():
    data = {
        "utterances": [],
        "intents": [{"id": 0, "regexp_full_match": ["\\d{3}"]}],
    }

    n_shots = 10
    dataset = sample_from_regex(Dataset.model_validate(data), n_shots)

    assert len(dataset.utterances) == n_shots
    assert all(re.fullmatch("\\d{3}", x.text) for x in dataset.utterances)
