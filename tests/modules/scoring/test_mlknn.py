import numpy as np

from autointent.context.data_handler import DataHandler, Dataset
from autointent.modules.scoring.mlknn.mlknn import MLKnnScorer
from tests.conftest import setup_environment


def test_base_mlknn(dataset):
    db_dir, dump_dir, logs_dir = setup_environment()

    test_dataset = Dataset.model_validate(
        {
            "utterances": [
                {
                    "text": "why is there a hold on my american saving bank account",
                    "label": [0, 1, 2],
                },
                {
                    "text": "i am nost sure why my account is blocked",
                    "label": [0, 2],
                },
            ],
        },
    )
    data_handler = DataHandler(dataset, test_dataset, force_multilabel=True)

    scorer = MLKnnScorer(db_dir=db_dir, k=3, embedder_name="sergeyzh/rubert-tiny-turbo")
    scorer.fit(data_handler.utterances_train, data_handler.labels_train)

    predictions = scorer.predict_labels(
        [
            "why is there a hold on my american saving bank account",
            "i am nost sure why my account is blocked",
            "why is there a hold on my capital one checking account",
            "i think my account is blocked but i do not know the reason",
            "can you tell me why is my bank account frozen",
        ]
    )
    assert (predictions == np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])).all()
