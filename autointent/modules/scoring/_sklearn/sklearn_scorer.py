import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import all_estimators
from typing_extensions import Self

from autointent import Context, Embedder
from autointent.custom_types import LabelType
from autointent.modules.abc import ScoringModule

logger = logging.getLogger(__name__)
AVAILABLE_CLASSIFIERS = {
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


class SklearnScorer(ScoringModule):
    """
    Scoring module for classification using sklearn classifiers with implemented predict_proba() method.

    This module uses embeddings generated from a transformer model to train
    chosen sklearn classifier for intent classification.

    :ivar name: Name of the scorer, defaults to "linear".
    """

    name = "sklearn"

    def __init__(
        self,
        embedder_name: str,
        clf_name: str,
        embedder_batch_size: int = 32,
        embedder_max_length: int | None = None,
        embedder_device: str = "cpu",
        embedder_use_cache: bool = True,
        clf_args: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the SklearnScorer.

        :param embedder_name: Name of the embedder model.
        :param clf_name: Name of the sklearn classifier to use.
        :param clf_args: dictionary with the chosen sklearn classifier arguments, defaults to {}.
        :param embedder_batch_size: Batch size for embedding generation, defaults to 32.
        :param embedder_max_length: Maximum sequence length for embedding, or None for default.
        :param embedder_device: Device to run operations on, e.g., "cpu" or "cuda".
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.embedder_name = embedder_name
        self.clf_name = clf_name
        self.clf_args = clf_args or {}
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_length = embedder_max_length
        self.embedder_device = embedder_device
        self.embedder_use_cache = embedder_use_cache

    @classmethod
    def from_context(
        cls,
        context: Context,
        clf_name: str = LogisticRegression.__name__,
        clf_args: dict[str, Any] | None = None,
        embedder_name: str | None = None,
    ) -> Self:
        """
        Create a SklearnScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param clf_name: Name of the sklearn classifier to use.
        :param clf_args: dictionary with the chosen sklearn classifier arguments, defaults to {}.
        :param embedder_name: Name of the embedder, or None to use the best embedder.
        :return: Initialized SklearnScorer instance.
        """
        if embedder_name is None:
            embedder_name = context.optimization_info.get_best_embedder()

        return cls(
            embedder_name=embedder_name,
            embedder_device=context.get_device(),
            embedder_batch_size=context.get_batch_size(),
            embedder_max_length=context.get_max_length(),
            embedder_use_cache=context.get_use_cache(),
            clf_name=clf_name,
            clf_args=clf_args,
        )

    def fit(
        self,
        utterances: list[str],
        labels: list[LabelType],
    ) -> None:
        """
        Train the chosen sklearn classifier.

        :param utterances: List of training utterances.
        :param labels: List of labels corresponding to the utterances.
        :raises ValueError: If the vector index mismatches the provided utterances.
        """
        self._multilabel = isinstance(labels[0], list)

        embedder = Embedder(
            device=self.embedder_device,
            model_name_or_path=self.embedder_name,
            batch_size=self.embedder_batch_size,
            max_length=self.embedder_max_length,
            use_cache=self.embedder_use_cache,
        )
        features = embedder.embed(utterances)
        if AVAILABLE_CLASSIFIERS.get(self.clf_name):
            base_clf = AVAILABLE_CLASSIFIERS[self.clf_name](**self.clf_args)
        else:
            msg = f"Class {self.clf_name} does not exist in sklearn or does not have predict_proba method"
            logger.error(msg)
            raise ValueError(msg)

        clf = MultiOutputClassifier(base_clf) if self._multilabel else base_clf

        clf.fit(features, labels)

        self._clf = clf
        self._embedder = embedder

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """
        Predict probabilities for the given utterances.

        :param utterances: List of query utterances.
        :return: Array of predicted probabilities for each class.
        """
        features = self._embedder.embed(utterances)
        probas = self._clf.predict_proba(features)
        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]
        return probas  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the embedder."""
        self._embedder.delete()
