import shutil
from unittest.mock import MagicMock

import numpy as np

from autointent.modules.embedding import LogRegEmbedding
from tests.conftest import setup_environment


def test_get_assets_returns_correct_artifact_for_logreg():
    module = LogRegEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo")
    artifact = module.get_assets()
    assert artifact.embedder_name == "sergeyzh/rubert-tiny-turbo"


def test_fit_trains_model():
    module = LogRegEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo")

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)

    assert module.classifier.coef_ is not None
    assert len(module.classifier.coef_) > 0
    assert module.label_encoder.classes_.tolist() == [0, 1]


def test_score_evaluates_model():
    module = LogRegEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo")

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)

    mock_context = MagicMock()
    mock_context.data_handler.test_utterances.return_value = ["hello", "goodbye"]
    mock_context.data_handler.test_labels.return_value = [0, 1]

    scores = module.score(mock_context, split="test")

    assert isinstance(scores, dict)


def test_dump_and_load_preserves_model_state():
    project_dir = setup_environment()
    module = LogRegEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo")

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)

    module.dump(project_dir)

    loaded_module = LogRegEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo")
    loaded_module.load(project_dir)
    epsilon = 1e-6

    assert np.allclose(loaded_module.classifier.coef_, module.classifier.coef_, atol=epsilon)
    assert np.allclose(loaded_module.classifier.intercept_, module.classifier.intercept_, atol=epsilon)
    assert np.array_equal(np.array(loaded_module.label_encoder.classes_), np.array(module.label_encoder.classes_))
    assert loaded_module.embedder_name == module.embedder_name

    shutil.rmtree(project_dir)
