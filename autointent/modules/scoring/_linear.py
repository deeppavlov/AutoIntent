"""LinearScorer class for linear classification."""

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier

from autointent import Context, Embedder
from autointent.custom_types import ListOfLabels
from autointent.modules.abc import ScoringModule
from autointent.schemas import EmbedderConfig, TaskTypeEnum


class LinearScorer(ScoringModule):
    """
    Scoring module for linear classification using logistic regression.

    This module uses embeddings generated from a transformer model to train a
    logistic regression classifier for intent classification.

    :ivar name: Name of the scorer, defaults to "linear".

    Example
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
        embedder_config: EmbedderConfig | str | dict[str, Any],
        cv: int = 3,
        n_jobs: int | None = None,
        seed: int = 0,
    ) -> None:
        """
        Initialize the LinearScorer.

        :param embedder_config: Config of the embedder model.
        :param cv: Number of cross-validation folds, defaults to 3.
        :param n_jobs: Number of parallel jobs for cross-validation, defaults to -1 (all CPUs).
        :param seed: Random seed for reproducibility, defaults to 0.
        """
        self.cv = cv
        self.n_jobs = n_jobs
        self.seed = seed
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_config: EmbedderConfig | str | None = None,
    ) -> "LinearScorer":
        """
        Create a LinearScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param embedder_config: Config of the embedder, or None to use the best embedder.
        :return: Initialized LinearScorer instance.
        """
        if embedder_config is None:
            embedder_config = context.optimization_info.get_best_embedder()

        return cls(
            embedder_config=embedder_config,
        )

    def get_embedder_config(self) -> dict[str, Any]:
        """
        Get the name of the embedder.

        :return: Embedder name.
        """
        return self.embedder_config.model_dump()

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        """
        Train the logistic regression classifier.

        :param utterances: List of training utterances.
        :param labels: List of labels corresponding to the utterances.
        :raises ValueError: If the vector index mismatches the provided utterances.
        """
        if hasattr(self, "_clf"):
            self.clear_cache()

        self._validate_task(labels)

        embedder = Embedder(
            self.embedder_config,
        )
        features = embedder.embed(utterances, TaskTypeEnum.classification)

        if self._multilabel:
            base_clf = LogisticRegression()
            clf = MultiOutputClassifier(base_clf)
        else:
            clf = LogisticRegressionCV(cv=self.cv, n_jobs=self.n_jobs, random_state=self.seed)

        clf.fit(features, labels)

        self._clf = clf
        self._embedder = embedder

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """
        Predict probabilities for the given utterances.

        :param utterances: List of query utterances.
        :return: Array of predicted probabilities for each class.
        """
        features = self._embedder.embed(utterances, TaskTypeEnum.classification)
        probas = self._clf.predict_proba(features)
        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]
        return probas  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the embedder."""
        self._embedder.clear_ram()
