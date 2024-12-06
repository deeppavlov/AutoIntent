import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import KNNScorer
from tests.conftest import setup_environment


def test_base_knn(dataset):
    db_dir, dump_dir, logs_dir = setup_environment()

    data_handler = DataHandler(dataset)

    scorer = KNNScorer(k=3, weights="distance", embedder_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir, device="cpu")

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
        predictions == np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    ).all()

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert "neighbors" in metadata[0]
