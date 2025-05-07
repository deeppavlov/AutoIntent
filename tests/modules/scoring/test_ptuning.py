import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import PTuningScorer


def test_ptuning_scorer_dump_load(dataset):
    """Test that PTuningScorer can be saved and loaded while preserving predictions."""
    data_handler = DataHandler(dataset)

    scorer_original = PTuningScorer(
        classification_model_config="prajjwal1/bert-tiny",
        num_train_epochs=1,
        batch_size=8,
        num_virtual_tokens=10,
        seed=42,
    )
    scorer_original.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my account",
        "why is my bank account frozen",
    ]

    predictions_before = scorer_original.predict(test_data)

    temp_dir_path = Path(tempfile.mkdtemp(prefix="ptuning_scorer_test_"))
    try:
        scorer_original.dump(str(temp_dir_path))

        scorer_loaded = PTuningScorer(
            classification_model_config="prajjwal1/bert-tiny",
            num_train_epochs=1,
            batch_size=8,
            num_virtual_tokens=10,
            seed=42,
        )
        scorer_loaded.load(str(temp_dir_path))

        assert hasattr(scorer_loaded, "_model")
        assert scorer_loaded._model is not None
        assert hasattr(scorer_loaded, "_tokenizer")
        assert scorer_loaded._tokenizer is not None

        predictions_after = scorer_loaded.predict(test_data)

        assert predictions_before.shape == predictions_after.shape
        np.testing.assert_allclose(predictions_before, predictions_after, atol=1e-6)

    finally:
        shutil.rmtree(temp_dir_path, ignore_errors=True)  # workaround for windows permission error


def test_ptuning_prediction(dataset):
    """Test that the transformer model can fit and make predictions."""
    data_handler = DataHandler(dataset)

    scorer = PTuningScorer(
        classification_model_config="prajjwal1/bert-tiny",
        num_train_epochs=1,
        batch_size=8,
        num_virtual_tokens=10,
        seed=42,
    )

    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
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


def test_ptuning_cache_clearing(dataset):
    """Test that the transformer model properly handles cache clearing."""
    data_handler = DataHandler(dataset)

    scorer = PTuningScorer(
        classification_model_config="prajjwal1/bert-tiny",
        num_train_epochs=1,
        batch_size=8,
        num_virtual_tokens=20,
        seed=42,
    )

    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = ["test text"]
    scorer.predict(test_data)
    scorer.clear_cache()

    assert not hasattr(scorer, "_model") or scorer._model is None
    assert not hasattr(scorer, "_tokenizer") or scorer._tokenizer is None

    with pytest.raises(RuntimeError):
        scorer.predict(test_data)
