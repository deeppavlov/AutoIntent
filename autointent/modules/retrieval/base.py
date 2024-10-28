from abc import ABC
from typing import Any

from autointent.modules.base import Module


class RetrievalModule(Module, ABC):
    def __init__(self, k: int, **kwargs: Any) -> None:  # noqa: ANN401,ARG002
        self.k = k
