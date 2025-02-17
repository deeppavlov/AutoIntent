"""Base class for embedding modules."""

from abc import ABC

from autointent.modules.abc import Module


class RegexModule(Module, ABC):
    """Base class for rule-based modules."""
