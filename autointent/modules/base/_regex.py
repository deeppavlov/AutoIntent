"""Base class for embedding modules."""

from abc import ABC
from typing import Any

from autointent.modules.base import BaseModule


class BaseRegex(BaseModule, ABC):
    """Base class for rule-based modules."""

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {}
