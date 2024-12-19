import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from autointent.modules.embedding import LogRegEmbedding
from tests.conftest import setup_environment


def test_get_assets_returns_correct_artifact_for_logreg():
    db_dir, dump_dir, logs_dir = setup_environment()
    module = LogRegEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir)
    artifact = module.get_assets()
    assert artifact.embedder_name == "sergeyzh/rubert-tiny-turbo"


def test_fit_trains_model():
    db_dir, dump_dir, logs_dir = setup_environment()
    module = LogRegEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir)

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)

    assert module.classifier.coef_ is not None
    assert len(module.classifier.coef_) > 0
    assert module.label_encoder.classes_.tolist() == [0, 1]


def test_score_evaluates_model():
    db_dir, dump_dir, logs_dir = setup_environment()
    module = LogRegEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir)

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)

    mock_context = MagicMock()
    mock_context.data_handler.test_utterances.return_value = ["hello", "goodbye"]
    mock_context.data_handler.test_labels.return_value = [0, 1]

    def mock_metric_fn(true_labels, predicted_labels):
        return sum(1 for t, p in zip(true_labels, predicted_labels[0], strict=False) if t == p) / len(true_labels)

    score = module.score(mock_context, split="test", metric_fn=mock_metric_fn)

    assert 0 <= score <= 1
    assert score > 0


def test_dump_and_load_preserves_model_state():
    db_dir, dump_dir, logs_dir = setup_environment()
    module = LogRegEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir)

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)

    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    module.dump(str(dump_path))

    loaded_module = LogRegEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir)
    loaded_module.load(str(dump_path))
    epsilon = 1e-6

    assert np.allclose(loaded_module.classifier.coef_, module.classifier.coef_, atol=epsilon)
    assert np.allclose(loaded_module.classifier.intercept_, module.classifier.intercept_, atol=epsilon)
    assert np.array_equal(np.array(loaded_module.label_encoder.classes_), np.array(module.label_encoder.classes_))
    assert loaded_module.embedder_name == module.embedder_name

    shutil.rmtree(dump_path)
