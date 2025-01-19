import pytest

from autointent.modules.regexp import RegExp


@pytest.mark.parametrize(
    ("partial_match", "expected_predictions"),
    [(".*", [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]), ("frozen", [[0], [0], [0], [0], [0, 1]])],
)
def test_base_regex(partial_match, expected_predictions):
    train_data = [
        {
            "id": 0,
            "name": "accept_reservations",
            "regexp_full_match": [".*"],
            "regexp_partial_match": [".*"],
        },
        {
            "id": 1,
            "name": "account_blocked",
            "regexp_partial_match": [partial_match],
        },
    ]

    matcher = RegExp()
    matcher.fit(train_data)

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]
    predictions = matcher.predict(test_data)
    assert predictions == expected_predictions

    predictions, metadata = matcher.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data) == len(metadata)

    assert "partial_matches" in metadata[0]
    assert "full_matches" in metadata[0]
