import tempfile

import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import SklearnScorer


def test_base_sklearn(dataset):
    data_handler = DataHandler(dataset)

    scorer = SklearnScorer(
        embedder_config="sergeyzh/rubert-tiny-turbo",
        clf_name="LogisticRegression",
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
    )

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
                [0.222, 0.287, 0.219, 0.271],
                [0.222, 0.287, 0.219, 0.271],
                [0.222, 0.287, 0.219, 0.271],
                [0.222, 0.287, 0.219, 0.271],
                [0.222, 0.287, 0.219, 0.271],
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
        new_scorer = SklearnScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_data)
        np.testing.assert_almost_equal(predictions, new_predictions, decimal=5)
