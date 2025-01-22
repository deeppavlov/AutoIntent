import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import SklearnScorer


def test_base_sklearn(dataset):
    data_handler = DataHandler(dataset)

    scorer = SklearnScorer(embedder_name="sergeyzh/rubert-tiny-turbo", clf_name="LogisticRegression")

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
                    0.23748632,
                    0.39067508,
                    0.2393372,
                    0.13250139,
                ],
                [0.23913757, 0.37610976, 0.24952359, 0.13522908],
                [
                    0.25714506,
                    0.34984371,
                    0.25495681,
                    0.13805442,
                ],
                [
                    0.2571957,
                    0.34850898,
                    0.25346288,
                    0.14083245,
                ],
                [
                    0.23885061,
                    0.41527567,
                    0.21830964,
                    0.12756408,
                ],
            ],
        ),
        predictions,
        decimal=2,
    )

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert metadata is None
