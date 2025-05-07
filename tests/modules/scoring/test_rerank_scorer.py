import tempfile

import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import RerankScorer


def test_base_rerank_scorer(dataset):
    data_handler = DataHandler(dataset)

    scorer = RerankScorer(
        k=3,
        weights="distance",
        embedder_config="sergeyzh/rubert-tiny-turbo",
        m=2,
        cross_encoder_config="cross-encoder/ms-marco-MiniLM-L6-v2",
    )

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]

    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))
    predictions = scorer.predict(test_data)
    assert (
        predictions
        == np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
    ).all()

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert "neighbors" in metadata[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        scorer.dump(temp_dir)
        del scorer
        new_scorer = RerankScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_data)
        assert np.allclose(predictions, new_predictions)
