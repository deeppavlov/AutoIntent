"""Base class for embedding modules."""

from abc import ABC

from autointent.modules.base import BaseModule


class BaseRegex(BaseModule, ABC):
    """Base class for rule-based modules."""
