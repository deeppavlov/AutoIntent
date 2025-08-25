import tempfile
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from autointent._wrappers.embedder import Embedder
from autointent.configs._transformers import EmbedderConfig, EmbedderFineTuningConfig, HFModelConfig
from autointent.context.data_handler import DataHandler


def test_finetune_dump_load(dataset, on_windows):
    """Test scenario: fine-tune -> dump -> load."""
    data_handler = DataHandler(dataset)

    # Setup config for fine-tuning
    hf_config = HFModelConfig(model_name="intfloat/multilingual-e5-small", batch_size=4, trust_remote_code=True)
    embedder_config = EmbedderConfig(
        **hf_config.model_dump(),
        default_prompt="Represent this text for retrieval:",
        similarity_fn_name="cosine",
        use_cache=False,
    )

    train_config = EmbedderFineTuningConfig(epoch_num=1, batch_size=4, early_stopping=False)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as temp_dir:
        temp_path = Path(temp_dir)

        # Step 1: Fine-tune the embedder
        embedder_original = Embedder(embedder_config)

        # Get embeddings before training
        test_utterances = ["Test sentence for embedding", "Another test utterance"]
        original_embeddings = embedder_original.embed(test_utterances)

        # Fine-tune the model
        embedder_original.train(
            utterances=data_handler.train_utterances(0),
            labels=data_handler.train_labels(0),
            config=train_config,
        )

        # Get embeddings after training (should be different)
        trained_embeddings = embedder_original.embed(test_utterances)

        # Verify that training changed the embeddings
        assert not np.allclose(
            original_embeddings, trained_embeddings, atol=1e-6
        ), "Embeddings should change after fine-tuning"

        # Step 2: Dump the fine-tuned embedder
        dump_path = temp_path / "fine_tuned_embedder"
        embedder_original.dump(dump_path)

        # Step 3: Load the embedder from dump
        embedder_loaded = Embedder.load(dump_path)

        # Test that loaded embedder produces same embeddings as fine-tuned one
        loaded_embeddings = embedder_loaded.embed(test_utterances)
        (
            np.testing.assert_allclose(trained_embeddings, loaded_embeddings, rtol=1e-5),
            "Loaded embedder should produce same embeddings as fine-tuned one",
        )


def test_dump_load_finetune(dataset, on_windows):
    """Test scenario: dump -> load -> fine-tune."""
    data_handler = DataHandler(dataset)

    # Setup config
    hf_config = HFModelConfig(model_name="intfloat/multilingual-e5-small", batch_size=4, trust_remote_code=True)
    embedder_config = EmbedderConfig(
        **hf_config.model_dump(),
        default_prompt="Represent this text for retrieval:",
        similarity_fn_name="cosine",
        use_cache=False,
    )

    train_config = EmbedderFineTuningConfig(epoch_num=1, batch_size=4, early_stopping=False)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as temp_dir:
        temp_path = Path(temp_dir)

        # Step 1: Create and dump original embedder (not fine-tuned)
        embedder_original = Embedder(embedder_config)
        test_utterances = ["Test sentence for embedding", "Another test utterance"]
        original_embeddings = embedder_original.embed(test_utterances)

        dump_path = temp_path / "original_embedder"
        embedder_original.dump(dump_path)

        # Step 2: Load the embedder from dump
        embedder_loaded = Embedder.load(dump_path)

        # Verify loaded embedder produces same embeddings
        loaded_embeddings = embedder_loaded.embed(test_utterances)
        (
            np.testing.assert_allclose(original_embeddings, loaded_embeddings, rtol=1e-5),
            "Loaded embedder should produce same embeddings as original",
        )

        # Step 3: Fine-tune the loaded embedder
        loaded_before_training = embedder_loaded.embed(test_utterances)

        embedder_loaded.train(
            utterances=data_handler.train_utterances(0),
            labels=data_handler.train_labels(0),
            config=train_config,
        )

        # Verify that training changed the embeddings
        loaded_after_training = embedder_loaded.embed(test_utterances)
        assert not np.allclose(
            loaded_before_training, loaded_after_training, atol=1e-6
        ), "Embeddings should change after fine-tuning the loaded model"


def test_load_from_disk_finetune_dump_load(dataset, on_windows):
    """Test scenario: load sentence transformer from disk -> fine-tune -> dump -> load."""
    data_handler = DataHandler(dataset)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as temp_dir:
        temp_path = Path(temp_dir)

        # Step 1: Save a sentence transformer model to disk
        model = SentenceTransformer("intfloat/multilingual-e5-small")
        model_disk_path = temp_path / "pretrained_model"
        model.save(str(model_disk_path))

        embedder_config = EmbedderConfig(
            model_name=str(model_disk_path),
            default_prompt="Represent this text for retrieval:",
            batch_size=4,
            use_cache=False,
            trust_remote_code=True,
        )

        embedder_from_disk = Embedder(embedder_config)
        test_utterances = ["Test sentence for embedding", "Another test utterance"]

        # Get embeddings before training
        embeddings_before_training = embedder_from_disk.embed(test_utterances)

        # Step 3: Fine-tune the embedder loaded from disk
        train_config = EmbedderFineTuningConfig(epoch_num=1, batch_size=4, early_stopping=False)
        embedder_from_disk.train(
            utterances=data_handler.train_utterances(0),
            labels=data_handler.train_labels(0),
            config=train_config,
        )

        # Get embeddings after training
        embeddings_after_training = embedder_from_disk.embed(test_utterances)

        # Verify that training changed the embeddings
        assert not np.allclose(
            embeddings_before_training, embeddings_after_training, atol=1e-6
        ), "Embeddings should change after fine-tuning"

        # Step 4: Dump the fine-tuned embedder
        dump_path = temp_path / "fine_tuned_from_disk"
        embedder_from_disk.dump(dump_path)

        # Step 5: Load the dumped fine-tuned embedder
        embedder_final = Embedder.load(dump_path)

        # Test that final loaded embedder produces same embeddings as fine-tuned one
        final_embeddings = embedder_final.embed(test_utterances)
        (
            np.testing.assert_allclose(embeddings_after_training, final_embeddings, rtol=1e-5),
            "Final loaded embedder should produce same embeddings as fine-tuned one",
        )


def test_embeddings_consistency_across_workflows(dataset, on_windows):
    """Test that different workflows produce consistent results when starting from same model."""
    data_handler = DataHandler(dataset)

    # Common config
    hf_config = HFModelConfig(model_name="intfloat/multilingual-e5-small", batch_size=4, trust_remote_code=True)
    embedder_config = EmbedderConfig(
        **hf_config.model_dump(),
        default_prompt="Represent this text for retrieval:",
        similarity_fn_name="cosine",
        use_cache=False,
    )
    train_config = EmbedderFineTuningConfig(epoch_num=1, batch_size=4, early_stopping=False)

    test_utterances = ["Test sentence for embedding"]
    train_data = {
        "utterances": data_handler.train_utterances(0)[:50],  # Same small subset
        "labels": data_handler.train_labels(0)[:50],
    }

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as temp_dir:
        temp_path = Path(temp_dir)

        # Workflow 1: Direct fine-tune
        embedder1 = Embedder(embedder_config)
        embedder1.train(**train_data, config=train_config)
        embeddings1 = embedder1.embed(test_utterances)

        # Workflow 2: Dump -> Load -> Fine-tune
        embedder2 = Embedder(embedder_config)
        dump_path2 = temp_path / "workflow2"
        embedder2.dump(dump_path2)
        embedder2_loaded = Embedder.load(dump_path2)
        embedder2_loaded.train(**train_data, config=train_config)
        embeddings2 = embedder2_loaded.embed(test_utterances)

        # Both workflows should produce similar results (allowing for minor training variance)
        # We use a more relaxed tolerance since different training runs can have slight variance
        (
            np.testing.assert_allclose(embeddings1, embeddings2, atol=1e-3),
            "Different workflows should produce similar embeddings",
        )


def test_multiple_dump_load_cycles_after_finetuning(dataset, on_windows):
    """Test that multiple dump/load cycles preserve fine-tuned model state."""
    data_handler = DataHandler(dataset)

    hf_config = HFModelConfig(model_name="intfloat/multilingual-e5-small", batch_size=4, trust_remote_code=True)
    embedder_config = EmbedderConfig(
        **hf_config.model_dump(),
        default_prompt="Represent this text for retrieval:",
        similarity_fn_name="cosine",
        use_cache=False,
    )

    train_config = EmbedderFineTuningConfig(epoch_num=1, batch_size=4, early_stopping=False)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as temp_dir:
        temp_path = Path(temp_dir)

        # Fine-tune original embedder
        embedder_original = Embedder(embedder_config)
        embedder_original.train(
            utterances=data_handler.train_utterances(0),
            labels=data_handler.train_labels(0),
            config=train_config,
        )

        test_utterances = ["Test consistency across multiple cycles"]
        fine_tuned_embeddings = embedder_original.embed(test_utterances)

        # First dump/load cycle
        dump_path1 = temp_path / "cycle1"
        embedder_original.dump(dump_path1)
        embedder_cycle1 = Embedder.load(dump_path1)
        cycle1_embeddings = embedder_cycle1.embed(test_utterances)

        # Second dump/load cycle
        dump_path2 = temp_path / "cycle2"
        embedder_cycle1.dump(dump_path2)
        embedder_cycle2 = Embedder.load(dump_path2)
        cycle2_embeddings = embedder_cycle2.embed(test_utterances)

        # All embeddings should be identical
        (
            np.testing.assert_allclose(fine_tuned_embeddings, cycle1_embeddings, rtol=1e-5),
            "First cycle should preserve embeddings",
        )
        (
            np.testing.assert_allclose(cycle1_embeddings, cycle2_embeddings, rtol=1e-5),
            "Second cycle should preserve embeddings",
        )
        (
            np.testing.assert_allclose(fine_tuned_embeddings, cycle2_embeddings, rtol=1e-5),
            "Multiple cycles should preserve fine-tuned embeddings",
        )
