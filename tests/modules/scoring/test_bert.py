import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import BertScorer


def test_bert_prediction(dataset):
    """Test that the transformer model can fit and make predictions."""
    data_handler = DataHandler(dataset)

    scorer = BertScorer(classification_model_config="prajjwal1/bert-tiny", num_train_epochs=1, batch_size=8)

    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]

    predictions = scorer.predict(test_data)

    # Verify prediction shape
    assert predictions.shape[0] == len(test_data)
    assert predictions.shape[1] == len(set(data_handler.train_labels(0)))

    # Verify predictions are probabilities
    assert 0.0 <= np.min(predictions) <= np.max(predictions) <= 1.0

    # Verify probabilities sum to 1 for multiclass
    if not scorer._multilabel:
        for pred_row in predictions:
            np.testing.assert_almost_equal(np.sum(pred_row), 1.0, decimal=5)

    # Test metadata function if available
    if hasattr(scorer, "predict_with_metadata"):
        predictions, metadata = scorer.predict_with_metadata(test_data)
        assert len(predictions) == len(test_data)
        assert metadata is None


def test_bert_cache_clearing(dataset):
    """Test that the transformer model properly handles cache clearing."""
    data_handler = DataHandler(dataset)

    scorer = BertScorer(classification_model_config="prajjwal1/bert-tiny", num_train_epochs=1, batch_size=8)

    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = ["test text"]

    # Should work before clearing cache
    scorer.predict(test_data)

    # Clear the cache
    scorer.clear_cache()

    # Verify model and tokenizer are removed
    assert not hasattr(scorer, "_model") or scorer._model is None
    assert not hasattr(scorer, "_tokenizer") or scorer._tokenizer is None

    # Should raise exception after clearing cache
    with pytest.raises(RuntimeError):
        scorer.predict(test_data)
