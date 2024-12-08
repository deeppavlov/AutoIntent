import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import LinearScorer
from tests.conftest import setup_environment


def test_base_linear(dataset):
    get_db_dir, dump_dir, logs_dir = setup_environment()

    data_handler = DataHandler(dataset)

    scorer = LinearScorer(embedder_name="sergeyzh/rubert-tiny-turbo", embedder_device="cpu")

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
                [
                    0.01828613,
                    0.93842264,
                    0.02633502,
                    0.01695622,
                ],
                [0.02662749, 0.89566195, 0.05008801, 0.02762255],
                [
                    0.08131153,
                    0.79191015,
                    0.07896874,
                    0.04780958,
                ],
                [
                    0.08382678,
                    0.77043132,
                    0.0826499,
                    0.063092,
                ],
                [
                    0.01482186,
                    0.9699848,
                    0.00757169,
                    0.00762165,
                ],
            ]
        ),
        predictions,
        decimal=2,
    )

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert metadata is None
