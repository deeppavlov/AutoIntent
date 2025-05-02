import numpy as np

from autointent.modules.embedding import LogregAimedEmbedding
from tests.conftest import setup_environment


def test_get_assets_returns_correct_artifact_for_logreg():
    module = LogregAimedEmbedding(embedder_config="sergeyzh/rubert-tiny-turbo")
    artifact = module.get_assets()
    assert artifact.config.model_name == "sergeyzh/rubert-tiny-turbo"


def test_fit_trains_model():
    module = LogregAimedEmbedding(embedder_config="sergeyzh/rubert-tiny-turbo")

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)

    assert module._classifier.coef_ is not None
    assert len(module._classifier.coef_) > 0
    assert module._label_encoder.classes_.tolist() == [0, 1]


def test_predict_evaluates_model():
    module = LogregAimedEmbedding(embedder_config="sergeyzh/rubert-tiny-turbo")

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)

    probas = module.predict(["hello", "bye"])

    assert len(probas) == 2
    assert probas[0][0] > probas[0][1]
    assert probas[1][1] > probas[1][0]


def test_dump_load():
    module = LogregAimedEmbedding(embedder_config="sergeyzh/rubert-tiny-turbo")
    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)
    predictions = module.predict(["hello", "bye"])

    dump_path = setup_environment()

    module.dump(dump_path)
    del module
    module = LogregAimedEmbedding.load(dump_path)
    predictions_loaded = module.predict(["hello", "bye"])
    assert np.allclose(predictions, predictions_loaded)
