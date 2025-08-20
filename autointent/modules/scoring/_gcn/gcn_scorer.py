from typing import Any

import numpy.typing as npt
import torch
from pydantic import PositiveInt
from sklearn.model_selection import train_test_split
from typing_extensions import Self

from autointent import Context, Embedder
from autointent.configs import (
    CrossEncoderConfig,
    EarlyStoppingConfig,
    EmbedderConfig,
    TaskTypeEnum,
    TorchTrainingConfig,
)
from autointent.custom_types import ListOfLabels
from autointent.modules.scoring._gcn.gcn_model import TextMLGCN
from autointent.modules.scoring._torch.base_scorer import BaseTorchTrainerScorer


class GCNScorer(BaseTorchTrainerScorer):
    name = "gcn"
    supports_multiclass = True
    supports_multilabel = True

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        label_embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        gcn_hidden_dims: list[int] | None = None,
        p_reweight: float = 0.2,
        tau_threshold: float = 0.4,
        num_train_epochs: PositiveInt = 10,
        batch_size: PositiveInt = 16,
        learning_rate: float = 1e-3,
        seed: int = 42,
        device: str | None = None,
        early_stopping_config: EarlyStoppingConfig | dict[str, Any] | None = None,
    ):
        if gcn_hidden_dims is None:
            gcn_hidden_dims = [1024]
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.label_embedder_config = EmbedderConfig.from_search_config(label_embedder_config)
        self.gcn_hidden_dims = gcn_hidden_dims
        self.p_reweight = p_reweight
        self.tau_threshold = tau_threshold
        self.torch_config = TorchTrainingConfig(
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
        if device is not None:
            self.torch_config.device = device
        self.early_stopping_config = EarlyStoppingConfig.from_search_config(early_stopping_config)

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        label_embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        gcn_hidden_dims: list[int] | None = None,
        p_reweight: float = 0.2,
        tau_threshold: float = 0.4,
        num_train_epochs: PositiveInt = 10,
        batch_size: PositiveInt = 16,
        learning_rate: float = 1e-3,
        seed: int = 42,
        early_stopping_config: EarlyStoppingConfig | dict[str, Any] | None = None,
    ) -> "GCNScorer":
        if embedder_config is None:
            embedder_config = context.resolve_embedder()
        if label_embedder_config is None:
            label_embedder_config = context.resolve_embedder()

        return cls(
            embedder_config=embedder_config,
            label_embedder_config=label_embedder_config,
            gcn_hidden_dims=gcn_hidden_dims,
            p_reweight=p_reweight,
            tau_threshold=tau_threshold,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=context.transformer_config.device,
            early_stopping_config=early_stopping_config,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {
            "embedder_config": self.embedder_config.model_dump(),
            "label_embedder_config": self.label_embedder_config.model_dump(),
        }

    def get_train_data(self, context: Context) -> tuple[list[str], ListOfLabels, list[str]]:
        descriptions = [intent.description or intent.name for intent in context.data_handler.dataset.intents]
        return (
            context.data_handler.train_utterances(0),
            context.data_handler.train_labels(0),
            descriptions,
        )

    def fit(self, utterances: list[str], labels: ListOfLabels, descriptions: list[str]) -> None:
        self._validate_task(labels)
        self._embedder = Embedder(self.embedder_config)
        self._label_embedder = Embedder(self.label_embedder_config)

        x_tensor = torch.tensor(self._embedder.embed(utterances, TaskTypeEnum.classification))
        y_tensor_dtype = torch.float if self._multilabel else torch.long
        y_tensor = torch.tensor(labels, dtype=y_tensor_dtype)

        self._label_embeddings = torch.tensor(
            self._label_embedder.embed(descriptions, TaskTypeEnum.classification)
        ).to(self.torch_config.device)

        self._model = TextMLGCN(
            num_classes=self._n_classes,
            bert_feature_dim=x_tensor.shape[1],
            label_embedding_dim=self._label_embeddings.shape[1],
            gcn_hidden_dims=self.gcn_hidden_dims,
            p_reweight=self.p_reweight,
            tau_threshold=self.tau_threshold,
        )

        y_corr_tensor = y_tensor if self._multilabel else torch.nn.functional.one_hot(y_tensor, self._n_classes)
        self._model.set_correlation_matrix(y_corr_tensor.float())

        if self.early_stopping_config.metric is not None:
            train_x, val_x, train_y, val_y = train_test_split(
                x_tensor,
                y_tensor,
                test_size=self.early_stopping_config.val_fraction,
                random_state=self.torch_config.seed,
            )
        else:
            train_x, val_x, train_y, val_y = x_tensor, None, y_tensor, None

        self._train_model(train_x, train_y, val_x, val_y, self._label_embeddings)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        if not hasattr(self, "_model"):
            msg = "Model is not trained. Call fit() first."
            raise RuntimeError(msg)
        x_tensor = torch.tensor(self._embedder.embed(utterances, TaskTypeEnum.classification))
        return self._predict_tensors(x_tensor, self._label_embeddings)

    def clear_cache(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_embedder"):
            self._embedder.clear_ram()
            del self._embedder
        if hasattr(self, "_label_embedder"):
            self._label_embedder.clear_ram()
            del self._label_embedder

    @classmethod
    def load(
        cls,
        path: str,
        embedder_config: EmbedderConfig | None = None,
        cross_encoder_config: CrossEncoderConfig | None = None,
    ) -> Self:
        instance = super().load(path, embedder_config, cross_encoder_config)
        if hasattr(instance, "_label_embeddings"):
            instance._label_embeddings = torch.tensor(instance._label_embeddings).to(instance.torch_config.device)
        return instance
