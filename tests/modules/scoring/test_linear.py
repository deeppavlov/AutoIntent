import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import LinearScorer


def test_base_linear(dataset):
    data_handler = DataHandler(dataset)

    scorer = LinearScorer(embedder_name="sergeyzh/rubert-tiny-turbo", embedder_device="cpu")

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
                [0.05998958, 0.74930755, 0.07045561, 0.12024726],
                [0.06299568, 0.60657246, 0.11424723, 0.21618463],
                [0.1285409, 0.53515583, 0.14137456, 0.19492871],
                [0.09999432, 0.3907234, 0.12208764, 0.38719464],
                [0.04322527, 0.85661047, 0.03667959, 0.06348467],
            ]
        ),
        predictions,
        decimal=2,
    )

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert metadata is None
