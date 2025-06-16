"""LinearScorer class for linear classification."""

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import PositiveInt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier

from autointent import Context, Embedder
from autointent.configs import EmbedderConfig, TaskTypeEnum
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer


class LinearScorer(BaseScorer):
    """Scoring module for linear classification using logistic regression.

    This module uses embeddings generated from a transformer model to train a
    logistic regression classifier for intent classification.

    Args:
        embedder_config: Config of the embedder model
        cv: Number of cross-validation folds, defaults to 3
        seed: Random seed for reproducibility, defaults to 0

    Example:
    --------
    .. testcode::

        from autointent.modules import LinearScorer
        scorer = LinearScorer(
            embedder_config="sergeyzh/rubert-tiny-turbo", cv=2
        )
        utterances = ["hello", "goodbye", "allo", "sayonara"]
        labels = [0, 1, 0, 1]
        scorer.fit(utterances, labels)
        test_utterances = ["hi", "bye"]
        probabilities = scorer.predict(test_utterances)
        print(probabilities)

    .. testoutput::

        [[0.50000032 0.49999968]
         [0.50000032 0.49999968]]

    """

    name = "linear"
    _multilabel: bool
    _clf: LogisticRegressionCV | MultiOutputClassifier
    _embedder: Embedder
    supports_multiclass = True
    supports_multilabel = True

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        cv: int = 3,
        seed: int = 0,
    ) -> None:
        self.cv = cv
        self.seed = seed
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)

        if self.cv < 0 or not isinstance(self.cv, int):
            msg = "`cv` argument of `LinearScorer` must be a positive int"
            raise ValueError(msg)

    @classmethod
    def from_context(
        cls,
        context: Context,
        cv: PositiveInt = 3,
        embedder_config: EmbedderConfig | str | None = None,
    ) -> "LinearScorer":
        """Create a LinearScorer instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            cv: Number of cross-validation folds, defaults to 3
            embedder_config: Config of the embedder, or None to use the best embedder
        """
        if embedder_config is None:
            embedder_config = context.resolve_embedder()

        return cls(
            cv=cv,
            embedder_config=embedder_config,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {"embedder_config": self.embedder_config.model_dump()}

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        """Train the logistic regression classifier.

        Args:
            utterances: List of training utterances
            labels: List of labels corresponding to the utterances

        Raises:
            ValueError: If the vector index mismatches the provided utterances
        """
        self._validate_task(labels)

        embedder = Embedder(
            self.embedder_config,
        )
        features = embedder.embed(utterances, TaskTypeEnum.classification)

        if self._multilabel:
            base_clf = LogisticRegression()
            clf = MultiOutputClassifier(base_clf)
        else:
            clf = LogisticRegressionCV(cv=self.cv, random_state=self.seed)

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
            self._embedder.clear_ram()
            delattr(self, "_clf")
