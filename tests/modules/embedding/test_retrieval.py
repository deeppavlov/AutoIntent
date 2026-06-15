from __future__ import annotations

from typing import TYPE_CHECKING

from autointent.configs import HashingVectorizerEmbeddingConfig
from autointent.modules.embedding import RetrievalAimedEmbedding
from tests.conftest import get_test_embedder_config

if TYPE_CHECKING:
    from pathlib import Path


def test_get_assets_returns_correct_artifact() -> None:
    module = RetrievalAimedEmbedding(k=5, embedder_config=get_test_embedder_config())
    artifact = module.get_assets()
    assert isinstance(artifact.config, HashingVectorizerEmbeddingConfig)
    assert artifact.config.n_features == 512


def test_dump_and_load_preserves_model_state(tmp_path: Path) -> None:
    module = RetrievalAimedEmbedding(k=5, embedder_config=get_test_embedder_config())

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)
    predictions = module.predict(utterances)

    module.dump(str(tmp_path))
    del module

    loaded_module = RetrievalAimedEmbedding.load(str(tmp_path))
    predictions_loaded = loaded_module.predict(utterances)
    assert predictions == predictions_loaded
