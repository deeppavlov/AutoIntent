import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import LinearScorer
from tests.conftest import setup_environment


def test_base_linear(dataset):
    get_db_dir, dump_dir, logs_dir = setup_environment()

    data_handler = DataHandler(dataset)

    scorer = LinearScorer(embedder_name="sergeyzh/rubert-tiny-turbo", device="cpu")

    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))
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
                [0.33332719, 0.33334283, 0.33332997],
                [0.33332507, 0.33334446, 0.33333046],
                [0.33332806, 0.33334067, 0.33333127],
                [0.33332788, 0.33334159, 0.33333053],
                [0.33332806, 0.33334418, 0.33332775],
            ],
        ),
        predictions,
        decimal=2,
    )

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert metadata is None
