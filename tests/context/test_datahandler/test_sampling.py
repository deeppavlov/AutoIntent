import re

import pytest

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
    intent_records = [
        {"intent_id": "greeting", "regexp_full_match": ["hello|hi", "hey there"], "sample_utterances": ["hello"]},
        {"intent_id": "farewell", "regexp_full_match": ["goodbye|bye"], "sample_utterances": ["goodbye"]},
    ]

    n_shots = 5
    result = sample_from_regex(intent_records, n_shots)

    assert len(result) == 2
    for intent in result:
        assert len(intent["sample_utterances"]) > 1
        for utterance in intent["sample_utterances"][1:]:  # Skip the original utterance
            assert any(re.fullmatch(pattern, utterance) for pattern in intent["regexp_full_match"])


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
    intent_records = [{"intent_id": "test", "regexp_full_match": ["\\d{3}"], "sample_utterances": []}]

    n_shots = 10
    result = sample_from_regex(intent_records, n_shots)

    assert len(result[0]["sample_utterances"]) == n_shots
    assert all(re.fullmatch("\\d{3}", x) for x in result[0]["sample_utterances"])
