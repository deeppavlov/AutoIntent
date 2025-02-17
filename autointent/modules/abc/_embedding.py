"""Base class for embedding modules."""

from abc import ABC

from autointent import Context
from autointent.custom_types import ListOfLabels
from autointent.modules.abc import BaseModule


class BaseEmbedding(BaseModule, ABC):
    """Base class for embedding modules."""

    def get_train_data(self, context: Context) -> tuple[list[str], ListOfLabels]:
        return (context.data_handler.train_utterances(0), context.data_handler.train_labels(0))  # type: ignore[return-value]
