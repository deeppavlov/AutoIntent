"""CatboostScorer class for CatBoost-based classification."""

from typing import Any

import numpy as np
import numpy.typing as npt
from catboost import CatBoostClassifier  # type: ignore[import-untyped]

from autointent import Context
from autointent.configs import EmbedderConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer

_BINARY_CLASS_COUNT = 2


class CatboostScorer(BaseScorer):
    """CatBoost scorer using embeddings as features.

    Args:
        embedder_config: Configuration for the embedder to use.
        iterations: Number of boosting iterations.
        learning_rate: Learning rate for CatBoost.
        loss_function: Loss function for CatBoost ('Logloss' for binary/multiclass, 'MultiLogloss' for multiclass).
        random_seed: Random seed for reproducibility.
        verbose: Whether to print CatBoost training progress.
        **catboost_kwargs: Additional arguments for CatBoostClassifier.
    """

    name = "catboost"
    supports_multiclass = True
    supports_multilabel = True

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        iterations: int = 100,
        learning_rate: float = 0.1,
        loss_function: str | None = None,
        random_seed: int = 0,
        verbose: bool = False,
        **catboost_kwargs: dict[str, Any],
    ) -> None:
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose
        self.catboost_kwargs = catboost_kwargs
        self._model: CatBoostClassifier

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        iterations: int = 100,
        learning_rate: float = 0.1,
        loss_function: str | None = None,
        random_seed: int = 0,
        verbose: bool = False,
        **catboost_kwargs: dict[str, Any],
    ) -> "CatboostScorer":
        """Create a CatboostScorer instance using a Context object."""
        if embedder_config is None:
            embedder_config = context.resolve_embedder()
        return cls(
            embedder_config=embedder_config,
            iterations=iterations,
            learning_rate=learning_rate,
            loss_function=loss_function,
            random_seed=random_seed,
            verbose=verbose,
            **catboost_kwargs,
        )

    def get_embedder_config(self) -> dict[str, Any]:
        """Return the configuration of the embedder model."""
        return self.embedder_config.model_dump()

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        """Train the CatBoost model using embeddings.

        Args:
            utterances: List of training utterances.
            labels: List of labels corresponding to the utterances.
        """
        if getattr(self, "_model", None) is not None:
            self.clear_cache()
        self._validate_task(labels)

        loss_function = self.loss_function
        if loss_function is None:
            loss_function = "MultiLogloss" if self._n_classes > _BINARY_CLASS_COUNT else "Logloss"
        self._model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            loss_function=loss_function,
            random_seed=self.random_seed,
            verbose=self.verbose,
            **self.catboost_kwargs,
        )
        fit_labels: Any = np.array(labels)
        self._model.fit(utterances, fit_labels)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """Predict probabilities for the given utterances using CatBoost.

        Args:
            utterances: List of query utterances.

        Returns:
            Array of predicted probabilities for each class (shape: [n_utterances, n_classes]).

        Raises:
            RuntimeError: If the model is not trained yet.
        """
        if getattr(self, "_model", None) is None:
            msg = "Model is not trained. Call fit() first."
            raise RuntimeError(msg)

        predictions = self._model.predict_proba(utterances)
        if not self._multilabel and self._n_classes == _BINARY_CLASS_COUNT and predictions.shape[1] == 1:
            predictions = np.hstack([1 - predictions, predictions])

        return np.asarray(predictions)

    def clear_cache(self) -> None:
        """Clear cached data in memory (model)."""
        if hasattr(self, "_model") and self._model is not None:
            del self._model
