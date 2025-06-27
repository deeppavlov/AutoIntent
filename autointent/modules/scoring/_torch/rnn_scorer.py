from typing import Any

from autointent import Context
from autointent.configs import EarlyStoppingConfig, TorchTrainingConfig, VocabConfig

from .base_scorer import BaseTorchScorer
from .rnn_model import TextRNN


class RNNScorer(BaseTorchScorer):
    """Scorer based on RNN model for text classification."""

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
