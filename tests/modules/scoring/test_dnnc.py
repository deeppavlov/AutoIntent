import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import DNNCScorer
from tests.conftest import setup_environment


@pytest.mark.parametrize(("train_head", "pred_score"), [(True, 1)])
def test_base_dnnc(dataset, train_head, pred_score):
    db_dir, dump_dir, logs_dir = setup_environment()

    data_handler = DataHandler(dataset)

    scorer = DNNCScorer(
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedder_name="sergeyzh/rubert-tiny-turbo",
        k=3,
        train_head=train_head,
        db_dir=db_dir,
        device="cpu",
    )

    scorer.fit(data_handler.train_utterances, data_handler.train_labels)
    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]
    predictions = scorer.predict(test_data)
    np.testing.assert_almost_equal(np.array([[0.0, pred_score, 0.0]] * len(test_data)), predictions, decimal=2)

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert "neighbors" in metadata[0]
    assert "scores" in metadata[0]

    scorer.clear_cache()
