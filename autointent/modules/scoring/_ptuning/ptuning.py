"""PTuningScorer class for ptuning-based classification."""

from typing import Any

import numpy.typing as npt

from autointent import Context
from autointent.configs import EmbedderConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer


class TokenizerConfig:
    """Configuration for tokenizer parameters."""

    def __init__(
        self,
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> None:
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation


class PTuningScorer(BaseScorer):
    """PEFT P-tuning scorer.

    Args:
        embedder_config: Config of the embedder model
        num_train_epochs: Number of training epochs, defaults to 3
        batch_size: Batch size for training, defaults to 8
        learning_rate: Learning rate for training, defaults to 5e-5
        seed: Random seed for reproducibility, defaults to 0
        tokenizer_config: Configuration for the tokenizer, defaults to None
        num_virtual_tokens: Number of virtual tokens for prompt tuning, defaults to 20
        prompt_tuning_init: Initialization method for prompt tuning, defaults to RANDOM

    Example:
    --------
    .. testcode::

    .. testoutput::

    """

    name = "ptuning"
    _multilabel: bool
    _model: Any
    _tokenizer: Any
    supports_multiclass = True
    supports_multilabel = True

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        tokenizer_config: TokenizerConfig | None = None,
        num_virtual_tokens: int = 20,
        prompt_tuning_init: str = "RANDOM",
    ) -> None:
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.tokenizer_config = tokenizer_config or TokenizerConfig()
        self.num_virtual_tokens = num_virtual_tokens
        self.prompt_tuning_init = prompt_tuning_init
        self._multilabel = False

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_config: EmbedderConfig | str | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        tokenizer_config: TokenizerConfig | None = None,
        num_virtual_tokens: int = 20,
        prompt_tuning_init: str = "RANDOM",
    ) -> "PTuningScorer":
        """Create a PTuningScorer instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            embedder_config: Config of the embedder, or None to use the best embedder
            num_train_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            seed: Random seed for reproducibility
            tokenizer_config: Configuration for the tokenizer
            num_virtual_tokens: Number of virtual tokens for prompt tuning
            prompt_tuning_init: Initialization method for prompt tuning
        """
        if embedder_config is None:
            embedder_config = context.resolve_embedder()

        return cls(
            embedder_config=embedder_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            tokenizer_config=tokenizer_config,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init=prompt_tuning_init,
        )

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        """Train the model using P-tuning with the PEFT library.

        Args:
            utterances: List of training utterances
            labels: List of labels corresponding to the utterances
        """

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """Predict probabilities for the given utterances.

        Args:
            utterances: List of query utterances

        Returns:
            Array of predicted probabilities for each class

        Raises:
            RuntimeError: If the model is not trained yet
        """
        if not hasattr(self, "_model") or not hasattr(self, "_tokenizer"):
            msg = "Model is not trained. Call fit() first."
            raise RuntimeError(msg)


    def clear_cache(self) -> None:
        """Clear cached data in memory used by the model and tokenizer."""
