import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules.scoring import MLKnnScorer


def test_base_mlknn(dataset):
    data_handler = DataHandler(dataset.to_multilabel())

    scorer = MLKnnScorer(embedder_config="sergeyzh/rubert-tiny-turbo", k=3)
    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]

    predictions = scorer.predict_labels(test_data)
    assert (
        predictions
        == np.array(
            [
                [
                    0,
                    1,
                    0,
                    0,
                ],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ]
        )
    ).all()

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert "neighbors" in metadata[0]
