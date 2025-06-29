"""Module for classification scoring using sklearn classifiers with predict_proba() method."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import all_estimators
from typing_extensions import Self

from autointent import Context, Embedder
from autointent.configs import EmbedderConfig, TaskTypeEnum
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer

logger = logging.getLogger(__name__)
AVAILABLE_CLASSIFIERS: dict[str, type[BaseEstimator]] = {
    name: class_
    for name, class_ in all_estimators(
        type_filter=[
            # remove transformer (e.g. TfidfTransformer) from the list of available classifiers
            "classifier",
            "regressor",
            "cluster",
        ]
    )
    if hasattr(class_, "predict_proba")
}


class SklearnScorer(BaseScorer):
    """Scoring module for classification using sklearn classifiers.

    This module uses embeddings generated from a transformer model to train
    chosen sklearn classifier for intent classification.

    Args:
        clf_name: Name of the sklearn classifier to use
        embedder_config: Config of the embedder model
        **clf_args: Arguments for the chosen sklearn classifier

    Examples:
        >>> from autointent.modules.scoring import SklearnScorer
        >>> utterances = ["hello", "how are you?"]
        >>> labels = [0, 1]
        >>> scorer = SklearnScorer(
        ...     clf_name="LogisticRegression",
        ...     embedder_config="sergeyzh/rubert-tiny-turbo",
        ... )
        >>> scorer.fit(utterances, labels)
        >>> test_utterances = ["hi", "what's up?"]
        >>> probabilities = scorer.predict(test_utterances)
    """

    name = "sklearn"
    supports_multilabel = True
    supports_multiclass = True

    def __init__(
        self,
        clf_name: str = "LogisticRegression",
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        **clf_args: dict[str, float | str | bool],
    ) -> None:
        """Initialize the SklearnScorer.

        Raises:
            ValueError: If the specified classifier doesn't exist or lacks predict_proba
        """
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.clf_name = clf_name

        clf_type = AVAILABLE_CLASSIFIERS.get(self.clf_name, None)
        if clf_type:
            self._base_clf = clf_type(**clf_args)
        else:
            msg = f"Class {self.clf_name} does not exist in sklearn or does not have predict_proba method"
            logger.error(msg)
            raise ValueError(msg)

    @property
    def trial_name(self) -> str:
        return f"sklearn_{self.clf_name}"

    @classmethod
    def from_context(
        cls,
        context: Context,
        clf_name: str = "LogisticRegression",
        embedder_config: EmbedderConfig | str | None = None,
        **clf_args: dict[str, float | str | bool],
    ) -> Self:
        """Create a SklearnScorer instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            clf_name: Name of the sklearn classifier to use
            embedder_config: Config of the embedder, or None to use the best embedder
            **clf_args: Arguments for the chosen sklearn classifier
        """
        if embedder_config is None:
            embedder_config = context.resolve_embedder()

        return cls(
            embedder_config=embedder_config,
            clf_name=clf_name,
            **clf_args,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {"embedder_config": self.embedder_config.model_dump()}

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        """Train the chosen sklearn classifier.

        Args:
            utterances: List of training utterances
            labels: List of labels corresponding to the utterances

        Raises:
            ValueError: If the vector index mismatches the provided utterances
        """
        self._validate_task(labels)

        embedder = Embedder(embedder_config=self.embedder_config)
        features = embedder.embed(utterances, TaskTypeEnum.classification)

        clf = MultiOutputClassifier(self._base_clf) if self._multilabel else self._base_clf

        clf.fit(features, labels)

        self._clf = clf
        self._embedder = embedder

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """Predict probabilities for the given utterances.

        Args:
            utterances: List of query utterances

        Returns:
            Array of predicted probabilities for each class
        """
        features = self._embedder.embed(utterances, TaskTypeEnum.classification)
        probas = self._clf.predict_proba(features)
        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]
        return probas  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the embedder."""
        if hasattr(self, "_clf"):
            delattr(self, "_clf")
        if hasattr(self, "_embedder"):
            self._embedder.delete()
