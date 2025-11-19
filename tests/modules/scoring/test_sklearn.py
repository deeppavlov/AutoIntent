import tempfile

import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import SklearnScorer
from tests.conftest import get_test_embedder_config


def test_base_sklearn(dataset):
    data_handler = DataHandler(dataset)

    scorer = SklearnScorer(
        embedder_config=get_test_embedder_config(),
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
                [0.19808616, 0.33850935, 0.20807189, 0.25533256],
                [0.21305655, 0.28760493, 0.22420657, 0.275132],
                [0.21481034, 0.2826606, 0.22563915, 0.27688998],
                [0.21779545, 0.27305433, 0.22861205, 0.2805381],
                [0.18922822, 0.3680897, 0.19876744, 0.2439147],
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
