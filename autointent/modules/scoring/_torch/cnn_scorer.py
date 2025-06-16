"""CNNScorer class for scoring."""

from typing import Any

from autointent import Context
from autointent.configs import TorchTrainingConfig, VocabConfig

from .base_scorer import BaseTorchScorer
from .cnn_model import TextCNN


class CNNScorer(BaseTorchScorer):
    """Convolutional Neural Network (CNN) scorer for intent classification."""

    name = "cnn"

    def __init__(
        self,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],  # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
        torch_config: TorchTrainingConfig | dict[str, Any] | None = None,
        vocab_config: VocabConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(torch_config=torch_config, vocab_config=vocab_config)

        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout = dropout

    @classmethod
    def from_context(
        cls,
        context: Context,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],  # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
        torch_config: TorchTrainingConfig | dict[str, Any] | None = None,
        vocab_config: VocabConfig | dict[str, Any] | None = None,
    ) -> "CNNScorer":
        torch_config = TorchTrainingConfig.from_search_config(torch_config)
        torch_config.report_to = context.logging_config.report_to  # type: ignore[assignment]
        return cls(
            embed_dim=embed_dim,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            dropout=dropout,
            vocab_config=vocab_config,
            torch_config=torch_config,
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
