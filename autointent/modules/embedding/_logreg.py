"""LogregAimedEmbedding class for a proxy optimzation of embedding."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import PositiveInt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

from autointent import Context, Embedder
from autointent.context.optimization_info import EmbeddingArtifact
from autointent.custom_types import ListOfLabels
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL
from autointent.modules.abc import BaseEmbedding
from autointent.schemas import EmbedderConfig, TaskTypeEnum


class LogregAimedEmbedding(BaseEmbedding):
    r"""
    Module for configuring embeddings optimized for linear classification.

    The main purpose of this module is to be used at embedding node for optimizing
    embedding configuration using its logreg classification quality as a sort of proxy metric.

    :ivar _classifier: The trained logistic regression model.
    :ivar _label_encoder: Label encoder for converting labels to numerical format.
    :ivar name: Name of the module, defaults to "logreg".

    Examples
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
        embedder_config: EmbedderConfig | str | dict[str, Any],
        cv: PositiveInt = 3,
    ) -> None:
        """
        Initialize the LogregAimedEmbedding.

        :param embedder_config: Config of the embedder used for creating embeddings.
        :param cv: the number of folds used in LogisticRegressionCV
        """
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.cv = cv

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_config: EmbedderConfig | str,
        cv: PositiveInt = 3,
    ) -> "LogregAimedEmbedding":
        """
        Create a LogregAimedEmbedding instance using a Context object.

        :param context: The context containing configurations and utilities.
        :param cv: the number of folds used in LogisticRegressionCV
        :param embedder_config: Config of the embedder to use.
        :return: Initialized LogregAimedEmbedding instance.
        """
        return cls(
            cv=cv,
            embedder_config=embedder_config,
        )

    def clear_cache(self) -> None:
        self._embedder.clear_ram()

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        """
        Train the logistic regression model using the provided utterances and labels.

        :param utterances: List of text data to index.
        :param labels: List of corresponding labels for the utterances.
        """
        if hasattr(self, "_embedder"):
            self.clear_cache()

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
        """
        Evaluate the embedding model using a specified metric function.

        :param context: The context containing test data and labels.
        :return: Computed metrics value for the test set or error code of metrics
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
        """
        Evaluate the embedding model using a specified metric function.

        :param context: The context containing test data and labels.
        :return: Computed metrics value for the test set or error code of metrics
        """
        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}

        metrics_calculated, _ = self.score_metrics_cv(chosen_metrics, context.data_handler.validation_iterator())
        return metrics_calculated

    def get_assets(self) -> EmbeddingArtifact:
        """
        Get the classifier artifacts for this module.

        :return: A EmbeddingArtifact object containing embedder information.
        """
        return EmbeddingArtifact(config=self.embedder_config)

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        embeddings = self._embedder.embed(utterances, TaskTypeEnum.classification)
        probas = self._classifier.predict_proba(embeddings)

        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]

        return probas  # type: ignore[no-any-return]
