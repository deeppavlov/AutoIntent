"""CNNScorer class for scoring."""

from typing import Any

from autointent import Context
from autointent.configs import EarlyStoppingConfig, TorchTrainingConfig, VocabConfig

from .base_scorer import BaseTorchScorer
from .cnn_model import TextCNN


class CNNScorer(BaseTorchScorer):
    """Convolutional Neural Network (CNN) scorer for intent classification.

    This module uses a CNN architecture to perform intent classification on text data.
    It builds a vocabulary from input text, converts text to indices, and trains a CNN
    model with multiple convolutional layers with different kernel sizes for feature
    extraction. Supports both multiclass and multilabel classification tasks.

    Args:
        embed_dim: Dimensionality of word embeddings (default: 128)
        kernel_sizes: List of kernel sizes for convolutional layers (default: [3, 4, 5])
        num_filters: Number of filters for each kernel size (default: 100)
        dropout: Dropout rate for regularization (default: 0.1)
        num_train_epochs: Number of training epochs (default: 3)
        batch_size: Batch size for training (default: 8)
        learning_rate: Learning rate for training (default: 5e-5)
        seed: Random seed for reproducibility (default: 42)
        device: Device for training ('cpu', 'cuda', etc.), auto-detected if None
        vocab_config: Configuration for vocabulary building
        early_stopping_config: Configuration for early stopping during training

    Example:
    --------
    .. testcode::

        from autointent.modules.scoring import CNNScorer

        # Initialize CNN scorer with custom parameters
        scorer = CNNScorer(
            embed_dim=16,
            kernel_sizes=[2, 3, 4],
            num_filters=3,
            dropout=0.2,
            num_train_epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            seed=42
        )

        # Training data
        utterances = ["This is great!", "I didn't like it", "Awesome product", "Poor quality"]
        labels = [1, 0, 1, 0]

        # Fit the model
        scorer.fit(utterances, labels)

        # Make predictions
        test_utterances = ["Good product", "Not worth it"]
        probabilities = scorer.predict(test_utterances)
    """

    name = "cnn"

    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],  # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 42,
        device: str | None = None,
        vocab_config: VocabConfig | dict[str, Any] | None = None,
        early_stopping_config: EarlyStoppingConfig | dict[str, Any] | None = None,
    ) -> None:
        torch_config = TorchTrainingConfig(
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
        if device is not None:
            torch_config.device = device
        super().__init__(
            torch_config=torch_config, vocab_config=vocab_config, early_stopping_config=early_stopping_config
        )

        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout = dropout

    @classmethod
    def from_context(  # noqa: PLR0913
        cls,
        context: Context,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],  # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 42,
        vocab_config: VocabConfig | dict[str, Any] | None = None,
        early_stopping_config: EarlyStoppingConfig | dict[str, Any] | None = None,
    ) -> "CNNScorer":
        return cls(
            embed_dim=embed_dim,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            dropout=dropout,
            vocab_config=vocab_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=context.transformer_config.device,
            early_stopping_config=early_stopping_config,
        )

    def _init_model(self) -> TextCNN:
        return TextCNN(
            n_classes=self._n_classes,
            embed_dim=self.embed_dim,
            kernel_sizes=self.kernel_sizes,
            num_filters=self.num_filters,
            dropout=self.dropout,
            vocab_config=self.vocab_config,
        )
