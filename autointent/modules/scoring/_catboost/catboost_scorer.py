"""CatBoostScorer class for CatBoost-based classification with switchable encoding."""

import logging
from enum import StrEnum
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from catboost import CatBoostClassifier  # type: ignore[import-untyped]

from autointent import Context, Embedder
from autointent.configs import EmbedderConfig, TaskTypeEnum
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer

BINARY_CLASS_THRESHOLD = 2

logger = logging.getLogger(__name__)


class FeaturesType(StrEnum):
    """Type of features used in CatBoostScorer."""

    TEXT = "text"
    EMBEDDING = "embedding"
    BOTH = "both"


class CatBoostScorer(BaseScorer):
    """CatBoost scorer using either external embeddings or CatBoost's own BoW encoding.

    Args:
        embedder_config: Config of the base transformer model (HFModelConfig, str, or dict)
            If None (default) the scorer relies on CatBoost's own Bag-of-Words encoding,
            otherwise the provided embedder is used.
        features_type: Type of features used in CatBoost. Can be one of:
            - "text": Use only text features (CatBoost's BoW encoding).
            - "embedding": Use only embedding features.
            - "both": Use both text and embedding features.
        use_embedding_features: If True, the model uses CatBoost `embedding_features` otherwise
            each number will be in separate column.
        loss_function: CatBoost loss function.  If None, an appropriate loss is
            chosen automatically from the task type.
        verbose: If True, CatBoost prints training progress.
        **catboost_kwargs: Any additional keyword arguments forwarded to
            :class:`catboost.CatBoostClassifier`.

    Example:
    -------
    .. testcode::

    from autointent.modules import CatBoostScorer


    scorer = CatBoostScorer(
        iterations=50,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="Accuracy",
        random_seed=42,
        verbose=False,
        features_type="embedding",  # or "text" or "both"
    )
    utterances = ["hello", "goodbye", "allo", "sayonara"]
    labels = [0, 1, 0, 1]
    scorer.fit(utterances, labels)
    test_utterances = ["hi", "bye"]
    probabilities = scorer.predict(test_utterances)
    print(probabilities)

    .. testoutput::

        [[0.41493207 0.58506793]
         [0.55036046 0.44963954]]

    """

    name = "catboost"
    supports_multiclass = True
    supports_multilabel = True

    _model: CatBoostClassifier

    encoder_features_types = (FeaturesType.EMBEDDING, FeaturesType.BOTH)

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        features_type: FeaturesType = FeaturesType.BOTH,
        use_embedding_features: bool = True,
        loss_function: str | None = None,
        verbose: bool = False,
        **catboost_kwargs: dict[str, Any],
    ) -> None:
        self.features_type = features_type
        self.use_embedding_features = use_embedding_features
        if features_type == FeaturesType.TEXT and use_embedding_features:
            msg = "Only catbooost text features will be used, `use_embedding_features` is ignored."
            logger.warning(msg)

        if self.features_type in self.encoder_features_types:
            self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
            self._embedder = Embedder(self.embedder_config)
        self.loss_function = loss_function
        self.verbose = verbose
        self.catboost_kwargs = catboost_kwargs or {}

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        features_type: FeaturesType = FeaturesType.BOTH,
        use_embedding_features: bool = True,
        loss_function: str | None = None,
        verbose: bool = False,
        **catboost_kwargs: dict[str, Any],
    ) -> "CatBoostScorer":
        if embedder_config is None:
            embedder_config = context.resolve_embedder()
        return cls(
            embedder_config=embedder_config,
            loss_function=loss_function,
            verbose=verbose,
            features_type=features_type,
            **catboost_kwargs,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {
            "embedder_config": self.embedder_config.model_dump()
            if self.features_type in self.encoder_features_types
            else None,
        }

    def _prepare_data_for_fit(
        self,
        utterances: list[str],
    ) -> pd.DataFrame:
        if self.features_type in self.encoder_features_types:
            encoded_utterances = self._embedder.embed(utterances, TaskTypeEnum.classification).tolist()
            if self.use_embedding_features:
                data = pd.DataFrame({"embedding": encoded_utterances})
            else:
                data = pd.DataFrame(encoded_utterances)
            if self.features_type == FeaturesType.BOTH:
                data["text"] = utterances
            return data
        return pd.DataFrame({"text": utterances})

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        if getattr(self, "_model", None) is not None:
            self.clear_cache()
        self._validate_task(labels)

        dataset = self._prepare_data_for_fit(utterances)

        default_loss = (
            "MultiLogloss"
            if self._multilabel
            else ("MultiClass" if self._n_classes > BINARY_CLASS_THRESHOLD else "Logloss")
        )

        extra_params = {}
        if self.features_type == FeaturesType.EMBEDDING:
            if self.use_embedding_features:  # to not raise error if embedding witout embedding_features
                extra_params["embedding_features"] = ["embedding"]
        elif self.features_type in {FeaturesType.TEXT, FeaturesType.BOTH}:
            extra_params["text_features"] = ["text"]
            if self.features_type == FeaturesType.BOTH and self.use_embedding_features:
                extra_params["embedding_features"] = ["embedding"]
        else:
            raise ValueError(f"Unsupported features type: {self.features_type}")
        self.catboost_kwargs.update(extra_params)

        self._model = CatBoostClassifier(
            loss_function=self.loss_function or default_loss,
            verbose=self.verbose,
            **self.catboost_kwargs,
        )
        self._model.fit(dataset, labels)

    def predict(self, utterances: list[str]) -> npt.NDArray[np.float64]:
        if getattr(self, "_model", None) is None:
            msg = "Model is not trained. Call fit() first."
            raise RuntimeError(msg)
        data = self._prepare_data_for_fit(utterances)
        return cast("npt.NDArray[np.float64]", self._model.predict_proba(data))

    def clear_cache(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_embedder"):
            del self._embedder
