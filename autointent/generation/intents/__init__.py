"""Generative methods for enriching intents' metadata."""

from ._description_generation import generate_descriptions
from ._prompt_scheme import PromptDescription

__all__ = ["PromptDescription", "generate_descriptions"]
