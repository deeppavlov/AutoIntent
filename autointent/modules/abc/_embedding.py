"""Base class for embedding modules."""

from abc import ABC

from autointent.modules.abc import Module


class EmbeddingModule(Module, ABC):
    """Base class for embedding modules."""

    def __init__(self, k: int) -> None:
        """
        Initialize embedding module.

        :param k: number of closest neighbors to consider during inference
        """
        self.k = k
