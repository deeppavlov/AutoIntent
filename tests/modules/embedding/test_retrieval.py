from pathlib import Path

from autointent.modules.embedding import RetrievalAimedEmbedding


def test_get_assets_returns_correct_artifact():
    module = RetrievalAimedEmbedding(k=5, embedder_config="sergeyzh/rubert-tiny-turbo")
    artifact = module.get_assets()
    assert artifact.config.model_name == "sergeyzh/rubert-tiny-turbo"


def test_dump_and_load_preserves_model_state(tmp_path: Path):
    module = RetrievalAimedEmbedding(k=5, embedder_config="sergeyzh/rubert-tiny-turbo")

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)
    predictions = module.predict(utterances)

    module.dump(tmp_path)
    del module

    loaded_module = RetrievalAimedEmbedding.load(tmp_path)
    predictions_loaded = loaded_module.predict(utterances)
    assert predictions == predictions_loaded
