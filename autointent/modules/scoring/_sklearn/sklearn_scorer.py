import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import all_estimators
from typing_extensions import Self

from autointent import Context, Embedder
from autointent.custom_types import ListOfLabels
from autointent.modules.abc import ScoringModule
from autointent.schemas import EmbedderConfig, TaskTypeEnum

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
    supports_multilabel = True
    supports_multiclass = True

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any],
        clf_name: str,
        clf_args: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the SklearnScorer.

        :param embedder_config: Config of the embedder model.
        :param clf_name: Name of the sklearn classifier to use.
        :param clf_args: dictionary with the chosen sklearn classifier arguments, defaults to {}.
        """
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.clf_name = clf_name
        self.clf_args = clf_args or {}

    @classmethod
    def from_context(
        cls,
        context: Context,
        clf_name: str = LogisticRegression.__name__,
        clf_args: dict[str, Any] | None = None,
        embedder_config: EmbedderConfig | str | None = None,
    ) -> Self:
        """
        Create a SklearnScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param clf_name: Name of the sklearn classifier to use.
        :param clf_args: dictionary with the chosen sklearn classifier arguments, defaults to {}.
        :param embedder_config: Config of the embedder, or None to use the best embedder.
        :return: Initialized SklearnScorer instance.
        """
        if embedder_config is None:
            embedder_config = context.optimization_info.get_best_embedder()

        return cls(
            embedder_config=embedder_config,
            clf_name=clf_name,
            clf_args=clf_args,
        )

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        """
        Train the chosen sklearn classifier.

        :param utterances: List of training utterances.
        :param labels: List of labels corresponding to the utterances.
        :raises ValueError: If the vector index mismatches the provided utterances.
        """
        if hasattr(self, "_clf"):
            self.clear_cache()

        self._validate_task(labels)

        embedder = Embedder(
            EmbedderConfig(
                model_name=self.embedder_config.model_name,
                device=self.embedder_config.device,
                batch_size=self.embedder_config.batch_size,
                max_length=self.embedder_config.max_length,
                use_cache=self.embedder_config.use_cache,
            )
        )
        features = embedder.embed(utterances, TaskTypeEnum.classification)
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
        features = self._embedder.embed(utterances, TaskTypeEnum.classification)
        probas = self._clf.predict_proba(features)
        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]
        return probas  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the embedder."""
        self._embedder.delete()
