from abc import ABC

from autointent.modules.base import Module


class RetrievalModule(Module, ABC):
    def __init__(self, k: int) -> None:
        self.k = k
