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


class LogregAimedEmbedding(EmbeddingModule):
    r"""
    Module for configuring embeddings optimized for linear classification.

    The main purpose of this module is to be used at embedding node for optimizing
    embedding configuration using its logreg classification quality as a sort of proxy metric.

    :ivar classifier: The trained logistic regression model.
    :ivar label_encoder: Label encoder for converting labels to numerical format.
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
    name = "logreg"
    supports_multiclass = True
    supports_multilabel = True
    supports_oos = False

    def __init__(
        self,
        embedder_name: str,
        cv: int = 3,
        embedder_device: str = "cpu",
        embedder_batch_size: int = 32,
        embedder_max_length: int | None = None,
        embedder_use_cache: bool = True,
    ) -> None:
        """
        Initialize the LogregAimedEmbedding.

        :param cv: the number of folds used in LogisticRegressionCV
        :param embedder_name: Name of the embedder used for creating embeddings.
        :param embedder_device: Device to run operations on, e.g., "cpu" or "cuda".
        :param batch_size: Batch size for embedding generation.
        :param max_length: Maximum sequence length for embeddings. None if not set.
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.embedder_name = embedder_name
        self.embedder_device = embedder_device
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_length = embedder_max_length
        self.embedder_use_cache = embedder_use_cache
        self.cv = cv

    @classmethod
    def from_context(
        cls,
        context: Context,
        cv: int,
        embedder_name: str,
    ) -> "LogregAimedEmbedding":
        """
        Create a LogregAimedEmbedding instance using a Context object.

        :param cv: the number of folds used in LogisticRegressionCV
        :param context: The context containing configurations and utilities.
        :param embedder_name: Name of the embedder to use.
        :return: Initialized LogregAimedEmbedding instance.
        """
        return cls(
            cv=cv,
            embedder_name=embedder_name,
            embedder_device=context.get_device(),
            embedder_batch_size=context.get_batch_size(),
            embedder_max_length=context.get_max_length(),
            embedder_use_cache=context.get_use_cache(),
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
            device=self.embedder_device,
            model_name_or_path=self.embedder_name,
            batch_size=self.embedder_batch_size,
            max_length=self.embedder_max_length,
            use_cache=self.embedder_use_cache,
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
        return RetrieverArtifact(embedder_name=self.embedder_name)

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        embeddings = self._embedder.embed(utterances)
        probas = self._classifier.predict_proba(embeddings)

        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]

        return probas  # type: ignore[no-any-return]
