from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from pydantic import PositiveInt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import Self

from autointent import Context, Embedder
from autointent.configs import CrossEncoderConfig, EmbedderConfig, TaskTypeEnum, TorchTrainingConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer
from autointent.modules.scoring._gcn.gcn_model import TextMLGCN


class GCNScorer(BaseScorer):
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
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {
            "embedder_config": self.embedder_config.model_dump(),
            "label_embedder_config": self.label_embedder_config.model_dump(),
        }

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        self._validate_task(labels)
        self._embedder = Embedder(self.embedder_config)
        self._label_embedder = Embedder(self.label_embedder_config)

        x_tensor = torch.tensor(self._embedder.embed(utterances, TaskTypeEnum.classification))
        y_tensor_dtype = torch.float if self._multilabel else torch.long
        y_tensor = torch.tensor(labels, dtype=y_tensor_dtype)

        intent_texts = [f"intent {i}" for i in range(self._n_classes)]
        self._label_embeddings = torch.tensor(
            self._label_embedder.embed(intent_texts, TaskTypeEnum.classification)
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

        criterion = nn.BCEWithLogitsLoss() if self._multilabel else nn.CrossEntropyLoss()
        self._train_model(x_tensor, y_tensor, criterion)

    def _train_model(self, train_x: torch.Tensor, train_y: torch.Tensor, criterion: nn.Module) -> None:
        train_dataset = TensorDataset(train_x, train_y)
        train_dataloader = DataLoader(train_dataset, batch_size=self.torch_config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.torch_config.learning_rate)

        self._model.to(self.torch_config.device)
        self._model.train()

        for _ in range(self.torch_config.num_train_epochs):
            for batch_x, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs = self._model(batch_x.to(self.torch_config.device), self._label_embeddings)
                loss = criterion(outputs, batch_y.to(self.torch_config.device))
                loss.backward()
                optimizer.step()

        self._model.eval()

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        if not hasattr(self, "_model"):
            raise RuntimeError("Model is not trained. Call fit() first.")

        x_tensor = torch.tensor(self._embedder.embed(utterances, TaskTypeEnum.classification))
        all_probs = []

        self._model.eval()
        with torch.no_grad():
            for i in range(0, len(x_tensor), self.torch_config.batch_size):
                batch_x = x_tensor[i : i + self.torch_config.batch_size].to(self.torch_config.device)
                outputs = self._model(batch_x, self._label_embeddings)
                if self._multilabel:
                    probs = torch.sigmoid(outputs).cpu().numpy()
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

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