"""Prompt strategies for LLM-based intent classification."""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from autointent.generation.chat_templates import Message, Role


class PromptStrategy(ABC):
    """Abstract base class for prompt strategies."""

    @abstractmethod
    def create_prompt(
        self,
        utterance: str,
        intent_names: List[str],
        train_utterances: List[str] | None = None,
        train_labels: List[str] | None = None,
    ) -> List[Message]:
        """Create a prompt for intent classification.

        Args:
            utterance: The utterance to classify
            intent_names: List of possible intent names
            train_utterances: Training utterances (for few-shot)
            train_labels: Training labels (for few-shot)

        Returns:
            List of messages forming the prompt
        """


class ZeroShotPromptStrategy(PromptStrategy):
    """Zero-shot prompting strategy.
    
    Uses only intent names without training examples to classify utterances.
    """

    def __init__(self, system_prompt: str | None = None):
        """Initialize zero-shot strategy.
        
        Args:
            system_prompt: Custom system prompt. If None, uses default.
        """
        self.system_prompt = system_prompt or self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for zero-shot classification."""
        return (
            "You are an intent classification assistant. Given an utterance and a list of possible intent names, "
            "classify the utterance by returning the most appropriate intent name. "
            "Return only the intent name, nothing else."
        )

    def create_prompt(
        self,
        utterance: str,
        intent_names: List[str],
        train_utterances: List[str] | None = None,
        train_labels: List[str] | None = None,
    ) -> List[Message]:
        """Create zero-shot prompt.

        Args:
            utterance: The utterance to classify
            intent_names: List of possible intent names
            train_utterances: Not used in zero-shot
            train_labels: Not used in zero-shot

        Returns:
            List of messages forming the prompt
        """
        intent_list = "\n".join([f"- {intent}" for intent in intent_names])
        
        user_message = (
            f"Classify the following utterance into one of these intents:\n\n"
            f"{intent_list}\n\n"
            f"Utterance: \"{utterance}\"\n\n"
            f"Intent:"
        )

        return [
            {"role": Role.SYSTEM, "content": self.system_prompt},
            {"role": Role.USER, "content": user_message}
        ]


class FewShotPromptStrategy(PromptStrategy):
    """Few-shot prompting strategy.
    
    Uses training examples to provide context for classification.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        max_examples_per_intent: int = 3,
        max_total_examples: int = 10,
        randomize_examples: bool = True,
    ):
        """Initialize few-shot strategy.
        
        Args:
            system_prompt: Custom system prompt. If None, uses default.
            max_examples_per_intent: Maximum examples per intent to include
            max_total_examples: Maximum total examples to include
            randomize_examples: Whether to randomize example selection
        """
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.max_examples_per_intent = max_examples_per_intent
        self.max_total_examples = max_total_examples
        self.randomize_examples = randomize_examples

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for few-shot classification."""
        return (
            "You are an intent classification assistant. Given an utterance and examples of different intents, "
            "classify the utterance by returning the most appropriate intent name. "
            "Use the provided examples to understand the pattern for each intent. "
            "Return only the intent name, nothing else."
        )

    def _select_examples(
        self,
        train_utterances: List[str],
        train_labels: List[str],
        intent_names: List[str],
    ) -> List[tuple[str, str]]:
        """Select examples for few-shot prompting.

        Args:
            train_utterances: Training utterances
            train_labels: Training labels
            intent_names: List of intent names

        Returns:
            List of (utterance, intent) tuples
        """
        intent_examples: Dict[str, List[str]] = {intent: [] for intent in intent_names}
        
        for utterance, label in zip(train_utterances, train_labels):
            if label in intent_examples:
                intent_examples[label].append(utterance)

        selected_examples = []
        for intent in intent_names:
            examples = intent_examples[intent]
            if not examples:
                continue
                
            if self.randomize_examples:
                examples = examples.copy()
                random.shuffle(examples)
            
            selected = examples[:self.max_examples_per_intent]
            selected_examples.extend([(ex, intent) for ex in selected])

        if len(selected_examples) > self.max_total_examples:
            if self.randomize_examples:
                random.shuffle(selected_examples)
            selected_examples = selected_examples[:self.max_total_examples]

        return selected_examples

    def create_prompt(
        self,
        utterance: str,
        intent_names: List[str],
        train_utterances: List[str] | None = None,
        train_labels: List[str] | None = None,
    ) -> List[Message]:
        """Create few-shot prompt.

        Args:
            utterance: The utterance to classify
            intent_names: List of possible intent names
            train_utterances: Training utterances for examples
            train_labels: Training labels for examples

        Returns:
            List of messages forming the prompt
        """
        if train_utterances is None or train_labels is None:
            raise ValueError("Few-shot strategy requires training examples")

        examples = self._select_examples(train_utterances, train_labels, intent_names)
        
        examples_text = ""
        if examples:
            examples_text = "Examples:\n"
            for ex_utterance, ex_intent in examples:
                examples_text += f'Utterance: "{ex_utterance}" → Intent: {ex_intent}\n'
            examples_text += "\n"

        intent_list = "\n".join([f"- {intent}" for intent in intent_names])
        
        user_message = (
            f"Classify the following utterance into one of these intents:\n\n"
            f"{intent_list}\n\n"
            f"{examples_text}"
            f"Now classify this utterance:\n"
            f'Utterance: "{utterance}"\n\n'
            f"Intent:"
        )

        return [
            {"role": Role.SYSTEM, "content": self.system_prompt},
            {"role": Role.USER, "content": user_message}
        ] 