from autointent.context.data_handler import DataHandler
from autointent._wrappers.embedder import Embedder
from autointent.configs._transformers import HFModelConfig, EmbedderConfig, EmbedderFineTuningConfig
import numpy as np

def test_model_updates_after_training(dataset):
    """Test that model weights actually change after training"""
    data_handler = DataHandler(dataset)

    hf_config = HFModelConfig(
    model_name="intfloat/multilingual-e5-small",
    batch_size=8,
    trust_remote_code=True
    )

    embedder_config = EmbedderConfig(
        **hf_config.model_dump(),
        default_prompt="Represent this text for retrieval:",
        query_prompt="Search query:",
        passage_prompt="Document:",
        similarity_fn_name="cosine",
        use_cache=False,
        freeze=False
    )

    train_config = EmbedderFineTuningConfig(
        epoch_num = 1
    )
    embedder = Embedder(embedder_config)
    embedder._load_model()

    original_weights = [
        param.data.detach().cpu().numpy().copy()
        for param in embedder.embedding_model.parameters()
        if param.requires_grad
    ]
    embedder.train(
        utterances=data_handler.train_utterances(0)[:10],
        labels=data_handler.train_labels(0)[:10],
        config=train_config
    )

    trained_weights = [
        param.data.detach().cpu().numpy().copy()
        for param in embedder.embedding_model.parameters()
        if param.requires_grad
    ]

    weights_changed = any(
        not np.allclose(orig, trained, atol=1e-6)
        for orig, trained in zip(original_weights, trained_weights)
    )
    
    assert weights_changed, "Model weights should change after training"
