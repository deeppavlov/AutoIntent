import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import BERTLoRAScorer


def test_lora_prediction(dataset):
    """Test that the transformer model can fit and make predictions."""
    data_handler = DataHandler(dataset)

    scorer = BERTLoRAScorer(transformer_config="prajjwal1/bert-tiny", num_train_epochs=1, batch_size=8)
    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am not sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]

    predictions = scorer.predict(test_data)

    assert predictions.shape[0] == len(test_data)
    assert predictions.shape[1] == len(set(data_handler.train_labels(0)))

    assert 0.0 <= np.min(predictions) <= np.max(predictions) <= 1.0

    if not scorer._multilabel:
        for pred_row in predictions:
            np.testing.assert_almost_equal(np.sum(pred_row), 1.0, decimal=5)

    if hasattr(scorer, "predict_with_metadata"):
        predictions, metadata = scorer.predict_with_metadata(test_data)
        assert len(predictions) == len(test_data)
        assert metadata is None


def test_bert_cache_clearing(dataset):
    """Test that the transformer model properly handles cache clearing."""
    data_handler = DataHandler(dataset)

    scorer = BERTLoRAScorer(transformer_config="prajjwal1/bert-tiny", num_train_epochs=1, batch_size=8)
    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = ["test text"]

    scorer.predict(test_data)
    scorer.clear_cache()

    assert not hasattr(scorer, "_model") or scorer._model is None
    assert not hasattr(scorer, "_tokenizer") or scorer._tokenizer is None

    with pytest.raises(RuntimeError):
        scorer.predict(test_data)
