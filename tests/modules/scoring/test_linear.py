import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import LinearScorer
from tests.conftest import setup_environment


def test_base_linear(dataset):
    get_db_dir, dump_dir, logs_dir = setup_environment()

    data_handler = DataHandler(dataset)

    scorer = LinearScorer("sergeyzh/rubert-tiny-turbo")

    scorer.fit(data_handler.utterances_train, data_handler.labels_train)
    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]
    predictions = scorer.predict(test_data)
    np.testing.assert_almost_equal(
        np.array(
            [
                [0.17929172, 0.59130114, 0.22940714],
                [0.15927979, 0.62363961, 0.2170806],
                [
                    0.20069508,
                    0.53883687,
                    0.26046804,
                ],
                [0.17557001, 0.61310582, 0.21132417],
                [
                    0.17911179,
                    0.63123131,
                    0.1896569,
                ],
            ]
        ),
        predictions,
        decimal=2,
    )
