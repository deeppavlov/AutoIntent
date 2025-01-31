"""Base class for chat templates for evolution augmentation."""

from abc import ABC, abstractmethod

from autointent.generation.utterances.schemas import Message
from autointent.schemas import Intent


class EvolutionChatTemplate(ABC):
    """Base class for chat templates for evolution augmentation."""

    @abstractmethod
    def __call__(self, utterance: str, intent_data: Intent) -> list[Message]:
        """Make a chat to complete by LLM."""
