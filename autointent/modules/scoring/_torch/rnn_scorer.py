from typing import Any

from autointent import Context
from autointent.configs import TorchTrainingConfig, VocabConfig

from .base import BaseTorchScorer
from .rnn_model import TextRNN


class RNNScorer(BaseTorchScorer):
    """Scorer based on RNN model for text classification."""

    name = "rnn"

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        torch_config: TorchTrainingConfig | dict[str, Any] | None = None,
        vocab_config: VocabConfig | dict[str, Any] | None = None,
    ) -> None:
        """Initialize the RNN scorer."""
        super().__init__(torch_config=torch_config, vocab_config=vocab_config)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

    @classmethod
    def from_context(
        cls,
        context: Context,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        torch_config: TorchTrainingConfig | dict[str, Any] | None = None,
        vocab_config: VocabConfig | dict[str, Any] | None = None,
    ) -> "RNNScorer":
        """Create a RNNScorer from context."""
        torch_config.report_to = context.logging_config.report_to

        return cls(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            torch_config=torch_config,
            vocab_config=vocab_config,
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
