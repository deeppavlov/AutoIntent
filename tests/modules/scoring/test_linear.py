import tempfile

import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import LinearScorer
from tests.conftest import get_test_embedder_config


def test_base_linear(dataset):
    data_handler = DataHandler(dataset)

    scorer = LinearScorer(embedder_config=get_test_embedder_config())

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
                [4.42261625e-03, 9.80002146e-01, 5.84225268e-03, 9.73298532e-03],
                [3.48457612e-02, 8.67882177e-01, 5.26664920e-02, 4.46055700e-02],
                [6.60129036e-02, 6.81724763e-01, 6.13724992e-02, 1.90889834e-01],
                [3.19191741e-01, 3.05030337e-01, 1.57439488e-01, 2.18338434e-01],
                [1.25137105e-04, 9.99343901e-01, 2.06237249e-04, 3.24724282e-04],
            ]
        ),
        predictions,
        decimal=2,
    )

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert metadata is None

    with tempfile.TemporaryDirectory() as temp_dir:
        scorer.dump(temp_dir)
        del scorer
        new_scorer = LinearScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_data)
        np.testing.assert_almost_equal(predictions, new_predictions, decimal=5)
