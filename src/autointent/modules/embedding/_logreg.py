"""LogregAimedEmbedding class for a proxy optimization of embedding."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import PositiveInt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

from autointent import Context, Embedder
from autointent.configs import EmbedderConfig, TaskTypeEnum
from autointent.context.optimization_info import EmbeddingArtifact
from autointent.custom_types import ListOfLabels
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL
from autointent.modules.base import BaseEmbedding


class LogregAimedEmbedding(BaseEmbedding):
    """Module for configuring embeddings optimized for linear classification.

    The main purpose of this module is to be used at embedding node for optimizing
    embedding configuration using its logreg classification quality as a sort of proxy metric.

    Args:
        embedder_config: Config of the embedder used for creating embeddings
        cv: Number of folds used in LogisticRegressionCV

    Examples:
    --------
    .. testcode::

        from autointent.modules.embedding import LogregAimedEmbedding
        utterances = ["bye", "how are you?", "good morning"]
        labels = [0, 1, 1]
        retrieval = LogregAimedEmbedding(
            embedder_config="sergeyzh/rubert-tiny-turbo",
            cv=2
        )
        retrieval.fit(utterances, labels)
    """

    _classifier: LogisticRegressionCV | MultiOutputClassifier
    _label_encoder: LabelEncoder | None
    name = "logreg_embedding"
    supports_multiclass = True
    supports_multilabel = True
    supports_oos = False

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        cv: PositiveInt = 3,
    ) -> None:
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.cv = cv

        if self.cv < 0 or not isinstance(self.cv, int):
            msg = "`cv` argument of `LogregAimedEmbedding` must be a positive int"
            raise ValueError(msg)

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_config: EmbedderConfig | str | None = None,
        cv: PositiveInt = 3,
    ) -> "LogregAimedEmbedding":
        """Create a LogregAimedEmbedding instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            cv: Number of folds used in LogisticRegressionCV
            embedder_config: Config of the embedder to use
        """
        return cls(
            cv=cv,
            embedder_config=embedder_config,
        )

    def clear_cache(self) -> None:
        """Clear embedder from memory."""
        if hasattr(self, "_embedder"):
            self._embedder.clear_ram()

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        """Train the logistic regression model using the provided utterances and labels.

        Args:
            utterances: List of text data to index
            labels: List of corresponding labels for the utterances
        """
        self._validate_task(labels)

        self._embedder = Embedder(
            self.embedder_config,
        )
        embeddings = self._embedder.embed(utterances, TaskTypeEnum.classification)

        if self._multilabel:
            self._label_encoder = None
            base_clf = LogisticRegression()
            self._classifier = MultiOutputClassifier(base_clf)
        else:
            self._label_encoder = LabelEncoder()
            labels = self._label_encoder.fit_transform(labels)
            self._classifier = LogisticRegressionCV(cv=self.cv)

        self._classifier.fit(embeddings, labels)

    def score_ho(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Evaluate the embedding model using specified metric functions.

        Args:
            context: Context containing test data and labels
            metrics: List of metric names to compute

        Returns:
            Dictionary of computed metric values for the test set
        """
        train_utterances, train_labels = self.get_train_data(context)
        self.fit(train_utterances, train_labels)

        val_utterances = context.data_handler.validation_utterances(0)
        val_labels = context.data_handler.validation_labels(0)

        probas = self.predict(val_utterances)
        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}

        return self.score_metrics_ho((val_labels, probas), chosen_metrics)

    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Evaluate the embedding model using specified metric functions.

        Args:
            context: Context containing test data and labels
            metrics: List of metric names to compute

        Returns:
            Dictionary of computed metric values for the test set
        """
        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}

        metrics_calculated, _ = self.score_metrics_cv(chosen_metrics, context.data_handler.validation_iterator())
        return metrics_calculated

    def get_assets(self) -> EmbeddingArtifact:
        """Get the classifier artifacts for this module.

        Returns:
            EmbeddingArtifact object containing embedder information
        """
        return EmbeddingArtifact(config=self.embedder_config)

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        """Predict probabilities for input utterances.

        Args:
            utterances: List of texts to predict probabilities for

        Returns:
            Array of predicted probabilities
        """
        embeddings = self._embedder.embed(utterances, TaskTypeEnum.classification)
        probas = self._classifier.predict_proba(embeddings)

        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]

        return probas  # type: ignore[no-any-return]
