"""CNNScorer class for scoring."""

from typing import Any

from autointent import Context
from autointent.configs import EarlyStoppingConfig, TorchTrainingConfig, VocabConfig

from .base_scorer import BaseTorchScorer
from .cnn_model import TextCNN


class CNNScorer(BaseTorchScorer):
    """Convolutional Neural Network (CNN) scorer for intent classification."""

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
