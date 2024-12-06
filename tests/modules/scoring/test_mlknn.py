import numpy as np
from datasets import Dataset as HFDataset

from autointent.context.data_handler import DataHandler
from autointent.custom_types import Split
from autointent.modules.scoring import MLKnnScorer
from tests.conftest import setup_environment


def test_base_mlknn(dataset):
    db_dir, dump_dir, logs_dir = setup_environment()

    dataset[Split.TEST] = HFDataset.from_list(
        [
            {
                "utterance": "why is there a hold on my american saving bank account",
                "label": [0, 1, 2],
            },
            {
                "utterance": "i am nost sure why my account is blocked",
                "label": [0, 2],
            },
        ],
    )

    data_handler = DataHandler(dataset, force_multilabel=True)

    scorer = MLKnnScorer(embedder_name="sergeyzh/rubert-tiny-turbo", k=3, db_dir=db_dir, device="cpu")
    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]

    predictions = scorer.predict_labels(test_data)
    assert (predictions == np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])).all()

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert "neighbors" in metadata[0]
