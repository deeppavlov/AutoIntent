import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import RNNScorer


def test_rnn_prediction(dataset):
    """Test that the RNN model can fit and make predictions."""
    data_handler = DataHandler(dataset)

    scorer = RNNScorer(embed_dim=8, hidden_dim=8, n_layers=1, num_train_epochs=1)
    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am not sure why my account is blocked",
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


def test_rnn_cache_clearing(dataset):
    """Test that the RNN model properly handles cache clearing."""
    data_handler = DataHandler(dataset)

    scorer = RNNScorer(embed_dim=8, hidden_dim=8, n_layers=1, num_train_epochs=1)
    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = ["test text"]

    # Should work before clearing cache
    scorer.predict(test_data)

    # Clear the cache
    scorer.clear_cache()

    # Verify model is removed
    assert not hasattr(scorer, "_model") or scorer._model is None

    # Should raise exception after clearing cache
    with pytest.raises(RuntimeError, match=r"Scorer is not trained.*"):
        scorer.predict(test_data)


def test_rnn_device(dataset):
    """Test RNN scorer with different device settings."""
    data_handler = DataHandler(dataset)

    # Force CPU
    scorer = RNNScorer(embed_dim=8, hidden_dim=8, n_layers=1, num_train_epochs=1, device="cpu")

    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = ["test account freeze"]
    scorer.predict(test_data)

    # Ensure model is on CPU
    assert next(scorer._model.parameters()).device.type == "cpu"


def test_rnn_scorer_dump_load(dataset):
    """Test that RNNScorer can be saved and loaded while preserving predictions."""
    data_handler = DataHandler(dataset)

    # Create and train scorer
    scorer_original = RNNScorer(embed_dim=8, hidden_dim=8, n_layers=1, num_train_epochs=1)
    scorer_original.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    # Test data
    test_data = [
        "why is there a hold on my account",
        "why is my bank account frozen",
    ]

    # Get predictions before saving
    predictions_before = scorer_original.predict(test_data)

    # Create temp directory and save model
    temp_dir_path = Path(tempfile.mkdtemp(prefix="rnn_scorer_test_"))
    try:
        # Save the model
        scorer_original.dump(str(temp_dir_path))

        # Create a new scorer and load saved model
        scorer_loaded = RNNScorer.load(str(temp_dir_path))

        # Verify model and vocabulary are loaded
        assert hasattr(scorer_loaded, "_model")
        assert scorer_loaded._model is not None
        assert scorer_loaded.vocab_config.vocab is not None

        # Get predictions after loading
        predictions_after = scorer_loaded.predict(test_data)

        # Verify predictions match
        assert predictions_before.shape == predictions_after.shape
        np.testing.assert_allclose(predictions_before, predictions_after, atol=1e-6)

    finally:
        # Clean up
        shutil.rmtree(temp_dir_path, ignore_errors=True)  # workaround for windows permission error
