import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import KNNScorer


def test_base_knn(setup_environment, dataset):
    db_dir, dump_dir, logs_dir = setup_environment

    data_handler = DataHandler(dataset)

    scorer = KNNScorer(k=3, weights="distance", model_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir())

    scorer.fit(data_handler.train_utterances, data_handler.train_labels)
    predictions = scorer.predict(
        [
            "why is there a hold on my american saving bank account",
            "i am nost sure why my account is blocked",
            "why is there a hold on my capital one checking account",
            "i think my account is blocked but i do not know the reason",
            "can you tell me why is my bank account frozen",
        ]
    )
    assert (
        predictions == np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    ).all()
