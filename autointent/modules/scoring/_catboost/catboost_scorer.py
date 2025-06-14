"""CatBoostScorer class for CatBoost-based classification with switchable encoding."""

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


class CatBoostScorer(BaseScorer):
    """CatBoost scorer using either external embeddings or CatBoost's own BoW encoding.

    Args:
        embedder_config: Config of the base transformer model (HFModelConfig, str, or dict)
            If None (default) the scorer relies on CatBoost's own Bag-of-Words encoding,
            otherwise the provided embedder is used.
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
    )
    utterances = ["hello", "goodbye", "allo", "sayonara"]
    labels = [0, 1, 0, 1]
    scorer.fit(utterances, labels)
    test_utterances = ["hi", "bye"]
    probabilities = scorer.predict(test_utterances)
    print(probabilities)

    .. testoutput::

        [[0.50525691 0.49474309]
         [0.50525691 0.49474309]]

    """

    name = "catboost"
    supports_multiclass = True
    supports_multilabel = True

    _model: CatBoostClassifier

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        use_embedder: bool = True,
        loss_function: str | None = None,
        verbose: bool = False,
        **catboost_kwargs: dict[str, Any],
    ) -> None:
        self.use_embedder = use_embedder
        if self.use_embedder:
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
        use_embedder: bool = True,
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
            use_embedder=use_embedder,
            **catboost_kwargs,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {
            "embedder_config": self.embedder_config.model_dump() if self.use_embedder else None,
        }

    def _prepare_embedding_dataset(
        self,
        utterances: list[str],
        labels: ListOfLabels | None,
    ) -> pd.DataFrame:
        encoded_utterances = self._embedder.embed(utterances, TaskTypeEnum.classification)
        return pd.DataFrame(
            {
                "text": utterances,
                "embedding": encoded_utterances.tolist(),
                "label": labels,
            }
        )

    def _prepare_text_dataset(
        self,
        utterances: list[str],
        labels: ListOfLabels | None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "text": utterances,
                "label": labels,
            }
        )

    def prepare_data_for_fit(
        self,
        utterances: list[str],
        labels: ListOfLabels | None,  # None for predict
    ) -> pd.DataFrame:
        if self.use_embedder:
            return self._prepare_embedding_dataset(utterances, labels)
        return self._prepare_text_dataset(utterances, labels)

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        if getattr(self, "_model", None) is not None:
            self.clear_cache()
        self._validate_task(labels)

        dataset = self.prepare_data_for_fit(utterances, labels)

        default_loss = (
            "MultiLogloss"
            if self._multilabel
            else ("MultiClass" if self._n_classes > BINARY_CLASS_THRESHOLD else "Logloss")
        )

        extra_params = {"text_features": ["text"]} if not self.use_embedder else {"embedding_features": ["embedding"]}
        self.catboost_kwargs.update(extra_params)

        self._model = CatBoostClassifier(
            loss_function=self.loss_function or default_loss,
            verbose=self.verbose,
            **self.catboost_kwargs,
        )
        self._model.fit(dataset)

    def predict(self, utterances: list[str]) -> npt.NDArray[np.float64]:
        if getattr(self, "_model", None) is None:
            msg = "Model is not trained. Call fit() first."
            raise RuntimeError(msg)
        data = self.prepare_data_for_fit(utterances, None)
        return cast("npt.NDArray[np.float64]", self._model.predict_proba(data))

    def clear_cache(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_embedder"):
            del self._embedder
