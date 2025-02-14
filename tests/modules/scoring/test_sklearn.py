import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import SklearnScorer


def test_base_sklearn(dataset):
    data_handler = DataHandler(dataset)

    scorer = SklearnScorer(embedder_config="sergeyzh/rubert-tiny-turbo", clf_name="LogisticRegression")

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
                [0.1853835, 0.37123936, 0.19039844, 0.2529787],
                [0.18256407, 0.34468965, 0.20265235, 0.27009393],
                [0.20398149, 0.32502411, 0.20803067, 0.26296374],
                [0.19607862, 0.31059503, 0.20248233, 0.29084402],
                [0.18350756, 0.40184831, 0.17685331, 0.23779081],
            ],
        ),
        predictions,
        decimal=2,
    )

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert metadata is None
