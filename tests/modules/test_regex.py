from autointent.modules import RegExp
from tests.conftest import setup_environment


def test_base_regex():
    db_dir, dump_dir, logs_dir = setup_environment()

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
                "regexp_full_match": [".*"],
                "regexp_partial_match": [".*"],
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
    assert predictions == [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
