import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import DNNCScorer


@pytest.mark.xfail(reason="This test is failing on windows, because have different score")
@pytest.mark.parametrize(("train_head", "pred_score"), [(True, 1), (False, 0.5)])
def test_base_dnnc(setup_environment, dataset, train_head, pred_score):
    db_dir, dump_dir, logs_dir = setup_environment

    data_handler = DataHandler(dataset)

    scorer = DNNCScorer("sergeyzh/rubert-tiny-turbo", k=3, train_head=train_head)

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
    scorer.clear_cache()
