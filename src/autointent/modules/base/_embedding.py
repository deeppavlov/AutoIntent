"""Base class for embedding modules."""

from abc import ABC
from typing import Any

from autointent import Context
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseModule


class BaseEmbedding(BaseModule, ABC):
    """Base class for embedding modules."""

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {}

    def get_train_data(self, context: Context) -> tuple[list[str], ListOfLabels]:
        """Get train data.

        Args:
            context: Context to get train data from

        Returns:
            Tuple of train utterances and train labels
        """
        return context.data_handler.train_utterances(0), context.data_handler.train_labels(0)  # type: ignore[return-value]
