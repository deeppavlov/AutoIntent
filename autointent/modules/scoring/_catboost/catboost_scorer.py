"""CatBoostScorer class for CatBoost-based classification with switchable encoding."""

import logging
from enum import Enum
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from catboost import CatBoostClassifier
from pydantic import PositiveInt

from autointent import Context, Embedder
from autointent.configs import EmbedderConfig, TaskTypeEnum
from autointent.custom_types import FloatFromZeroToOne, ListOfLabels
from autointent.modules.base import BaseScorer

logger = logging.getLogger(__name__)


class FeaturesType(str, Enum):
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

        val_fraction: fraction of training data used for early stopping. Set to None to disaple early stopping.
                Note: early stopping is not supported with multilabel classification.

        early_stopping_rounds: number of iterations without metric increasing waiting for early stopping.
                Ignored when ``val_fraction`` is ``None``.

        **catboost_kwargs: Any additional keyword arguments forwarded to
                :class:`catboost.CatBoostClassifier`. Please refer to
                `catboost's documentation <https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier>`_

    Example:
    --------

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
        val_fraction: float | None = 0.2,
        early_stopping_rounds: int = 100,
        iterations: int = 1000,
        depth: int = 6,
        **catboost_kwargs: dict[str, Any],
    ) -> None:
        self.val_fraction = val_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.iterations = iterations
        self.depth = depth
        self.features_type = features_type
        self.use_embedding_features = use_embedding_features
        if features_type == FeaturesType.TEXT and use_embedding_features:
            msg = "Only catbooost text features will be used, `use_embedding_features` is ignored."
            logger.warning(msg)

        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
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
        val_fraction: FloatFromZeroToOne | None = 0.2,
        early_stopping_rounds: PositiveInt = 100,
        iterations: PositiveInt = 1000,
        depth: PositiveInt = 6,
        **catboost_kwargs: dict[str, Any],
    ) -> "CatBoostScorer":
        if embedder_config is None:
            embedder_config = context.resolve_embedder()
        return cls(
            embedder_config=embedder_config,
            loss_function=loss_function,
            verbose=verbose,
            features_type=features_type,
            use_embedding_features=use_embedding_features,
            val_fraction=val_fraction,
            early_stopping_rounds=early_stopping_rounds,
            iterations=iterations,
            depth=depth,
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
                data = pd.DataFrame(np.array(encoded_utterances))
            if self.features_type == FeaturesType.BOTH:
                data["text"] = utterances
        else:
            data = pd.DataFrame({"text": utterances})

        return data

    def get_extra_params(self) -> dict[str, Any]:
        extra_params = {}
        if self.features_type == FeaturesType.EMBEDDING:
            if self.use_embedding_features:  # to not raise error if embedding without embedding_features
                extra_params["embedding_features"] = ["embedding"]
        elif self.features_type in {FeaturesType.TEXT, FeaturesType.BOTH}:
            extra_params["text_features"] = ["text"]
            if self.features_type == FeaturesType.BOTH and self.use_embedding_features:
                extra_params["embedding_features"] = ["embedding"]
        else:
            msg = f"Unsupported features type: {self.features_type}"
            raise ValueError(msg)
        return extra_params

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        self._validate_task(labels)

        if self.features_type in self.encoder_features_types:
            self._embedder = Embedder(self.embedder_config)

        dataset = self._prepare_data_for_fit(utterances)

        default_loss = (
            "MultiLogloss" if self._multilabel else ("MultiClass" if self._n_classes > 2 else "Logloss")  # noqa: PLR2004
        )

        if self._multilabel:
            self.val_fraction = None
            msg = "Disabling early stopping in CatBoostClassifier as it is not supported with multi-label task."
            logger.warning(msg)

        self._model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            loss_function=self.loss_function or default_loss,
            verbose=self.verbose,
            allow_writing_files=False,
            eval_fraction=self.val_fraction,
            **self.catboost_kwargs,
            **self.get_extra_params(),
        )
        self._model.fit(
            dataset, labels, early_stopping_rounds=self.early_stopping_rounds if self.val_fraction is not None else None
        )

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
