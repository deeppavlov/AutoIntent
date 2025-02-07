"""LogregAimedEmbedding class for a proxy optimzation of embedding."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

from autointent import Context, Embedder
from autointent.context.optimization_info import RetrieverArtifact
from autointent.custom_types import ListOfLabels
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL
from autointent.modules.abc import EmbeddingModule
from autointent.schemas._schemas import EmbedderConfig


class LogregAimedEmbedding(EmbeddingModule):
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
            embedder_name="sergeyzh/rubert-tiny-turbo",
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
        embedder_config: EmbedderConfig | str,
        cv: int = 3,
    ) -> None:
        """
        Initialize the LogregAimedEmbedding.

        :param embedder_config: Name of the embedder used for creating embeddings.
        :param cv: the number of folds used in LogisticRegressionCV
        """
        if isinstance(embedder_config, dict):
            embedder_config = EmbedderConfig(**embedder_config)
        elif isinstance(embedder_config, str):
            embedder_config = EmbedderConfig(model_name=embedder_config)
        self.embedder_config = embedder_config
        self.cv = cv

    @classmethod
    def from_context(
        cls,
        context: Context,
        cv: int = 3,
        embedder_config: EmbedderConfig | str,
    ) -> "LogregAimedEmbedding":
        """
        Create a LogregAimedEmbedding instance using a Context object.

        :param context: The context containing configurations and utilities.
        :param cv: the number of folds used in LogisticRegressionCV
        :param embedder_config: Name of the embedder to use.
        :return: Initialized LogregAimedEmbedding instance.
        """
        return cls(
            cv=cv,
            embedder_config=embedder_config,
        )

    def clear_cache(self) -> None:
        pass

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        """
        Train the logistic regression model using the provided utterances and labels.

        :param utterances: List of text data to index.
        :param labels: List of corresponding labels for the utterances.
        """
        self._validate_task(labels)

        self._embedder = Embedder(
            self.embedder_config,
        )
        embeddings = self._embedder.embed(utterances)

        if self._multilabel:
            self._label_encoder = None
            base_clf = LogisticRegression()
            self._classifier = MultiOutputClassifier(base_clf)
        else:
            self._label_encoder = LabelEncoder()
            labels = self._label_encoder.fit_transform(labels)
            self._classifier = LogisticRegressionCV(cv=self.cv)

        self._classifier.fit(embeddings, labels)

    def score(self, context: Context, split: Literal["validation", "test"], metrics: list[str]) -> dict[str, float]:
        """
        Evaluate the embedding model using a specified metric function.

        :param context: The context containing test data and labels.
        :param split: Target split
        :return: Computed metrics value for the test set or error code of metrics
        """
        if split == "validation":
            utterances = context.data_handler.validation_utterances(0)
            labels = context.data_handler.validation_labels(0)
        elif split == "test":
            utterances = context.data_handler.test_utterances()
            labels = context.data_handler.test_labels()
        else:
            message = f"Invalid split '{split}' provided. Expected one of 'validation', or 'test'."
            raise ValueError(message)

        probas = self.predict(utterances)
        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}
        return self.score_metrics((labels, probas), chosen_metrics)

    def get_assets(self) -> RetrieverArtifact:
        """
        Get the classifier artifacts for this module.

        :return: A RetrieverArtifact object containing embedder information.
        """
        return RetrieverArtifact(config=self.embedder_config)

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        embeddings = self._embedder.embed(utterances)
        probas = self._classifier.predict_proba(embeddings)

        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]

        return probas  # type: ignore[no-any-return]
