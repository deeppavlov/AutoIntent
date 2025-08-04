from typing import Any

from autointent import Context
from autointent.configs import EarlyStoppingConfig, TorchTrainingConfig, VocabConfig

from .base_scorer import BaseTorchScorer
from .rnn_model import TextRNN


class RNNScorer(BaseTorchScorer):
    """Recurrent Neural Network (RNN) scorer for intent classification.

    This module uses an RNN architecture to perform intent classification on text data.
    It builds a vocabulary from input text, converts text to indices, and trains an RNN
    model with multiple recurrent layers for sequential feature extraction. The RNN
    processes text sequentially, making it well-suited for capturing temporal dependencies
    in language. Supports both multiclass and multilabel classification tasks.

    Args:
        embed_dim: Dimensionality of word embeddings (default: 128)
        hidden_dim: Dimensionality of hidden states in RNN layers (default: 512)
        n_layers: Number of recurrent layers (default: 2)
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

        from autointent.modules.scoring import RNNScorer

        # Initialize RNN scorer with custom parameters
        scorer = RNNScorer(
            embed_dim=16,
            hidden_dim=16,
            n_layers=1,
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

    name = "rnn"

    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 42,
        device: str | None = None,
        vocab_config: VocabConfig | dict[str, Any] | None = None,
        early_stopping_config: EarlyStoppingConfig | dict[str, Any] | None = None,
    ) -> None:
        """Initialize the RNN scorer."""
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
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

    @classmethod
    def from_context(  # noqa: PLR0913
        cls,
        context: Context,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 42,
        vocab_config: VocabConfig | dict[str, Any] | None = None,
        early_stopping_config: EarlyStoppingConfig | dict[str, Any] | None = None,
    ) -> "RNNScorer":
        """Create a RNNScorer from context."""
        return cls(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=context.transformer_config.device,
            vocab_config=vocab_config,
            early_stopping_config=early_stopping_config,
        )

    def _init_model(self) -> TextRNN:
        return TextRNN(
            n_classes=self._n_classes,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            vocab_config=self.vocab_config,
        )
