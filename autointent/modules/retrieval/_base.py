"""Base class for retrieval modules."""

from abc import ABC

from autointent.modules._base import Module


class RetrievalModule(Module, ABC):
    """Base class for retrieval modules."""

    def __init__(self, k: int) -> None:
        """
        Initialize retrieval module.

        :param k: number of closest neighbors to consider during inference
        """
        self.k = k
