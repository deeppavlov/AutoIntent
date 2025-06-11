"""LLM-based scorer for intent classification."""

import logging
from typing import Any, Dict, List, Literal

import numpy as np
from numpy.typing import NDArray

from autointent import Context
from autointent.configs import LLMConfig
from autointent.context.optimization_info import ScorerArtifact
from autointent.custom_types import ListOfLabels
from autointent.generation import Generator
from autointent.metrics import SCORING_METRICS_MULTICLASS
from autointent.modules.base import BaseScorer

from .prompt_strategies import FewShotPromptStrategy, PromptStrategy, ZeroShotPromptStrategy

logger = logging.getLogger(__name__)


class LLMScorer(BaseScorer):
    """LLM-based intent classification scorer.

    This scorer uses Large Language Models for intent classification with support for
    both zero-shot and few-shot prompting strategies.

    Args:
        llm_config: Configuration for the LLM
        strategy: Prompting strategy ('zero_shot' or 'few_shot')
        prompt_strategy: Custom prompt strategy instance
        max_examples_per_intent: Maximum examples per intent for few-shot (default: 3)
        max_total_examples: Maximum total examples for few-shot (default: 10)
        randomize_examples: Whether to randomize example selection (default: True)
        fallback_intent: Fallback intent name when LLM fails to classify (default: "unknown")
    """

    name = "llm"
    supports_multiclass = True
    supports_multilabel = False
    supports_oos = True

    def __init__(
        self,
        llm_config: LLMConfig | str | Dict[str, Any] | None = None,
        strategy: Literal["zero_shot", "few_shot"] = "zero_shot",
        prompt_strategy: PromptStrategy | None = None,
        max_examples_per_intent: int = 3,
        max_total_examples: int = 10,
        randomize_examples: bool = True,
        fallback_intent: str = "unknown",
    ):
        """Initialize the LLM scorer."""
        self.llm_config = LLMConfig.from_search_config(llm_config)
        self.strategy = strategy
        self.fallback_intent = fallback_intent
        
        if prompt_strategy is not None:
            self.prompt_strategy = prompt_strategy
        elif strategy == "zero_shot":
            self.prompt_strategy = ZeroShotPromptStrategy()
        elif strategy == "few_shot":
            self.prompt_strategy = FewShotPromptStrategy(
                max_examples_per_intent=max_examples_per_intent,
                max_total_examples=max_total_examples,
                randomize_examples=randomize_examples,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self._generator: Generator | None = None
        
        self._train_utterances: List[str] | None = None
        self._train_labels: List[str] | None = None
        self._intent_names: List[str] | None = None
        self._label_to_intent: Dict[int, str] | None = None
        self._intent_to_label: Dict[str, int] | None = None

    @classmethod
    def from_context(
        cls,
        context: Context,
        llm_config: LLMConfig | str | Dict[str, Any] | None = None,
        strategy: Literal["zero_shot", "few_shot"] = "zero_shot",
        prompt_strategy: PromptStrategy | None = None,
        max_examples_per_intent: int = 3,
        max_total_examples: int = 10,
        randomize_examples: bool = True,
        fallback_intent: str = "unknown",
    ) -> "LLMScorer":
        """Create LLMScorer instance from context.

        Args:
            context: Context containing configurations
            llm_config: LLM configuration
            strategy: Prompting strategy
            prompt_strategy: Custom prompt strategy
            max_examples_per_intent: Max examples per intent for few-shot
            max_total_examples: Max total examples for few-shot
            randomize_examples: Whether to randomize examples
            fallback_intent: Fallback intent name

        Returns:
            LLMScorer instance
        """
        return cls(
            llm_config=llm_config,
            strategy=strategy,
            prompt_strategy=prompt_strategy,
            max_examples_per_intent=max_examples_per_intent,
            max_total_examples=max_total_examples,
            randomize_examples=randomize_examples,
            fallback_intent=fallback_intent,
        )

    def get_implicit_initialization_params(self) -> Dict[str, Any]:
        """Get implicit initialization parameters."""
        return {
            "llm_config": self.llm_config.model_dump(),
            "strategy": self.strategy,
            "fallback_intent": self.fallback_intent,
        }

    def fit(
        self,
        utterances: List[str],
        labels: ListOfLabels,
        intent_names: List[str] | None = None,
    ) -> None:
        """Fit the scorer with training data.

        Args:
            utterances: Training utterances
            labels: Training labels (can be integers or strings)
            intent_names: List of intent names (if labels are integers)
        """
        self._validate_task(labels)
        
        self._generator = Generator(
            base_url=self.llm_config.base_url,
            model_name=self.llm_config.model_name,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
            **self.llm_config.generation_params,
        )

        if isinstance(labels[0], int):
            if intent_names is None:
                unique_labels = sorted(set(labels))  # type: ignore[arg-type]
                intent_names = [f"intent_{i}" for i in unique_labels]
            
            self._label_to_intent = {i: intent_names[i] for i in range(len(intent_names))}
            self._intent_to_label = {intent: i for i, intent in enumerate(intent_names)}
            train_labels = [self._label_to_intent[label] for label in labels]  # type: ignore[index]
        else:
            unique_labels = sorted(set(labels))  # type: ignore[arg-type]
            self._intent_to_label = {intent: i for i, intent in enumerate(unique_labels)}
            self._label_to_intent = {i: intent for intent, i in self._intent_to_label.items()}
            train_labels = list(labels)  # type: ignore[assignment]
            intent_names = unique_labels

        self._train_utterances = utterances
        self._train_labels = train_labels
        self._intent_names = intent_names
        self._n_classes = len(intent_names)

    def predict(self, utterances: List[str]) -> NDArray[np.float64]:
        """Predict intent probabilities for utterances.

        Args:
            utterances: List of utterances to classify

        Returns:
            Array of probabilities for each intent
        """
        if self._generator is None:
            raise RuntimeError("Model not fitted. Call fit() before predict().")
        
        if self._intent_names is None:
            raise RuntimeError("Intent names not initialized. Call fit() before predict().")

        predictions = []
        
        for utterance in utterances:
            try:
                messages = self.prompt_strategy.create_prompt(
                    utterance=utterance,
                    intent_names=self._intent_names,
                    train_utterances=self._train_utterances,
                    train_labels=self._train_labels,
                )
                
                response = self._generator.get_chat_completion(messages)
                
                predicted_intent = self._parse_response(response)
                
                probabilities = self._intent_to_probabilities(predicted_intent)
                predictions.append(probabilities)
                
            except Exception as e:
                logger.warning(f"Error predicting intent for utterance '{utterance}': {e}")
                fallback_probs = np.ones(self._n_classes) / self._n_classes
                predictions.append(fallback_probs)

        return np.array(predictions, dtype=np.float64)

    def _parse_response(self, response: str) -> str:
        """Parse LLM response to extract intent name.

        Args:
            response: Raw LLM response

        Returns:
            Parsed intent name
        """
        if not response:
            return self.fallback_intent
        
        response = response.strip()
        
        if self._intent_names and response in self._intent_names:
            return response
        
        if self._intent_names:
            for intent in self._intent_names:
                if intent.lower() in response.lower():
                    return intent
        
        logger.warning(f"Could not parse intent from response: '{response}'")
        return self.fallback_intent

    def _intent_to_probabilities(self, predicted_intent: str) -> NDArray[np.float64]:
        """Convert predicted intent to probability distribution.

        Args:
            predicted_intent: Predicted intent name

        Returns:
            Probability distribution over all intents
        """
        if not self._intent_to_label or not self._intent_names:
            raise RuntimeError("Intent mappings not initialized")
        
        probabilities = np.zeros(self._n_classes, dtype=np.float64)
        
        if predicted_intent in self._intent_to_label:
            probabilities[self._intent_to_label[predicted_intent]] = 1.0
        else:
            probabilities.fill(1.0 / self._n_classes)
        
        return probabilities

    def clear_cache(self) -> None:
        """Clear cached data."""
        pass

    def get_assets(self) -> ScorerArtifact:
        """Get scorer artifacts.

        Returns:
            ScorerArtifact containing scorer information
        """
        return ScorerArtifact(
            module_name=self.name,
            module_config={
                "llm_config": self.llm_config.model_dump(),
                "strategy": self.strategy,
                "fallback_intent": self.fallback_intent,
            },
        )

    def get_train_data(self, context: Context) -> tuple[List[str], ListOfLabels, List[str]]:
        """Get training data from context.

        Args:
            context: Context containing training data

        Returns:
            Tuple of utterances, labels, and intent names
        """
        utterances = context.data_handler.train_utterances(0)
        labels = context.data_handler.train_labels(0)
        intent_names = context.data_handler.intent_names
        return utterances, labels, intent_names

    def score_ho(self, context: Context, metrics: List[str]) -> Dict[str, float]:
        """Score using holdout validation.

        Args:
            context: Context containing test data
            metrics: List of metrics to compute

        Returns:
            Dictionary of metric scores
        """
        train_utterances, train_labels, intent_names = self.get_train_data(context)
        
        self.fit(train_utterances, train_labels, intent_names)
        
        test_utterances = context.data_handler.test_utterances(0)
        test_labels = context.data_handler.test_labels(0)
        
        predictions = self.predict(test_utterances)
        
        metrics_dict = {name: SCORING_METRICS_MULTICLASS[name] for name in metrics}
        return self.score_metrics_ho((test_labels, predictions), metrics_dict)

    def score_cv(self, context: Context, metrics: List[str]) -> Dict[str, float]:
        """Score using cross-validation.

        Args:
            context: Context containing data
            metrics: List of metrics to compute

        Returns:
            Dictionary of averaged metric scores
        """
        intent_names = context.data_handler.intent_names
        cv_iterator = context.data_handler.cv_iterator()
        metrics_dict = {name: SCORING_METRICS_MULTICLASS[name] for name in metrics}
        cv_scores, _ = self.score_metrics_cv(
            metrics_dict=metrics_dict,
            cv_iterator=cv_iterator,
            intent_names=intent_names,
        )
        
        return cv_scores
