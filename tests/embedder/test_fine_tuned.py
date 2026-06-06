import numpy as np
import pytest

from autointent._wrappers.embedder.sentence_transformers import SentenceTransformerEmbeddingBackend
from autointent.configs import EmbedderFineTuningConfig
from autointent.context.data_handler import DataHandler
from tests.conftest import tiny_sentence_transformer_config


def test_model_updates_after_training(dataset):
    """Test that model weights actually change after training"""
    pytest.importorskip("accelerate", reason="Accelerate library is required for this test")

    data_handler = DataHandler(dataset)

    embedder_config = tiny_sentence_transformer_config(
        batch_size=8,
        trust_remote_code=True,
        default_prompt="Represent this text for retrieval:",
        query_prompt="Search query:",
        passage_prompt="Document:",
        similarity_fn_name="cosine",
    )

    train_config = EmbedderFineTuningConfig(epoch_num=3, batch_size=8)

    # Test with backend directly for fine-tuning specific functionality
    backend = SentenceTransformerEmbeddingBackend(embedder_config)
    backend._model = backend._load_model()

    for param in backend._model.parameters():
        assert param.requires_grad, "All trainable parameters should have requires_grad=True"

    original_weights = [
        param.data.detach().cpu().numpy().copy() for param in backend._model.parameters() if param.requires_grad
    ]

    backend.train(
        utterances=data_handler.train_utterances(0)[:1000],
        labels=data_handler.train_labels(0)[:1000],
        config=train_config,
    )

    trained_weights = [
        param.data.detach().cpu().numpy().copy() for param in backend._model.parameters() if param.requires_grad
    ]

    weights_changed = any(
        not np.allclose(orig, trained, atol=1e-6)
        for orig, trained in zip(original_weights, trained_weights, strict=True)
    )
    assert weights_changed, "Model weights should change after training"
