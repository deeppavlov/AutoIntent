import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules.scoring._cnn import CNNScorer


def test_cnn_prediction(dataset):
    """Test that the CNN model can fit and make predictions."""
    data_handler = DataHandler(dataset)

    scorer = CNNScorer(
        max_seq_length=50,
        num_train_epochs=1,
        batch_size=8,
        learning_rate=5e-5,
        embed_dim=128,
        kernel_sizes=(3, 4, 5),
        num_filters=100,
        dropout=0.1,
    )
    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my account",
        "i am not sure why my account is blocked",
        "why is there a hold on my checking account",
        "i think my account is blocked",
        "can you tell me why is my account frozen",
    ]

    predictions = scorer.predict(test_data)

    assert predictions.shape[0] == len(test_data)
    assert predictions.shape[1] == len(set(data_handler.train_labels(0)))

    # Проверяем что предсказания в диапазоне [0, 1]
    assert 0.0 <= np.min(predictions) <= np.max(predictions) <= 1.0

    # Для мультиклассовой классификации сумма предсказаний должна быть ~1.0
    if not scorer._multilabel:
        for pred_row in predictions:
            np.testing.assert_almost_equal(np.sum(pred_row), 1.0, decimal=5)

    # Проверяем работу predict_with_metadata если метод существует
    if hasattr(scorer, "predict_with_metadata"):
        predictions, metadata = scorer.predict_with_metadata(test_data)
        assert len(predictions) == len(test_data)
        assert metadata is None


def test_cnn_cache_clearing(dataset):
    """Test that the CNN model properly handles cache clearing."""
    data_handler = DataHandler(dataset)

    scorer = CNNScorer(max_seq_length=50, num_train_epochs=1, batch_size=8, learning_rate=5e-5)
    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = ["test text"]

    # Первое предсказание
    scorer.predict(test_data)

    # Очистка кэша
    scorer.clear_cache()

    # Проверяем что модель очищена
    assert not hasattr(scorer, "_model") or scorer._model is None

    # После очистки кэша предсказания должны вызывать ошибку
    with pytest.raises(ValueError, match=r"CNNScorer is not trained\. Call fit\(\) first\."):
        scorer.predict(test_data)


def test_cnn_scorer_dump_load(dataset):
    """Test that BERTLoRAScorer can be saved and loaded while preserving predictions."""
    data_handler = DataHandler(dataset)

    # Create and train scorer
    scorer = CNNScorer(max_seq_length=50, num_train_epochs=1, batch_size=8, learning_rate=5e-5)
    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    # Test data
    test_data = [
        "why is there a hold on my account",
        "why is my bank account frozen",
    ]

    # Get predictions before saving
    predictions_before = scorer.predict(test_data)

    # Create temp directory and save model
    temp_dir_path = Path(tempfile.mkdtemp(prefix="lora_scorer_test_"))
    try:
        # Save the model
        scorer.dump(str(temp_dir_path))

        # Create a new scorer and load saved model
        scorer_loaded = CNNScorer.load(str(temp_dir_path))

        # Verify model is loaded
        assert hasattr(scorer_loaded, "_model")
        assert scorer_loaded._model is not None

        # Get predictions after loading
        predictions_after = scorer_loaded.predict(test_data)

        # Verify predictions match
        assert predictions_before.shape == predictions_after.shape
        np.testing.assert_allclose(predictions_before, predictions_after, atol=1e-6)

    finally:
        # Clean up
        shutil.rmtree(temp_dir_path, ignore_errors=True)  # workaround for windows permission error
