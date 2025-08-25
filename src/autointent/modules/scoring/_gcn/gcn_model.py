import json
from pathlib import Path
from typing import cast

import torch
from pydantic import BaseModel
from torch import nn
from typing_extensions import Self

from autointent._utils import detect_device
from autointent._wrappers import BaseTorchModule


class GCNModelDumpMetadata(BaseModel):
    num_classes: int
    bert_feature_dim: int
    label_embedding_dim: int
    gcn_hidden_dims: list[int]
    p_reweight: float
    tau_threshold: float


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, adj_matrix: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        support = self.linear(features)
        return torch.matmul(adj_matrix, support)


class TextMLGCN(BaseTorchModule):
    _metadata_dict_name = "metadata.json"
    _state_dict_name = "state_dict.pt"
    correlation_matrix: torch.Tensor
    label_embeddings: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        bert_feature_dim: int,
        label_embedding_dim: int,
        gcn_hidden_dims: list[int],
        p_reweight: float,
        tau_threshold: float,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.p_reweight = p_reweight
        self.tau_threshold = tau_threshold
        self.bert_feature_dim = bert_feature_dim
        self.label_embedding_dim = label_embedding_dim
        self.gcn_hidden_dims = gcn_hidden_dims

        gcn_layers = []
        activation_layers: list[nn.LeakyReLU] = []

        in_dim = label_embedding_dim
        for hidden_dim in gcn_hidden_dims:
            gcn_layers.append(GCNLayer(in_dim, hidden_dim))
            activation_layers.append(nn.LeakyReLU(0.2))
            in_dim = hidden_dim

        gcn_layers.append(GCNLayer(in_dim, bert_feature_dim))
        activation_layers.append(nn.LeakyReLU(0.2))

        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.activations = nn.ModuleList(activation_layers)

        self.register_buffer("correlation_matrix", torch.zeros(num_classes, num_classes))
        self.register_buffer("label_embeddings", torch.zeros(num_classes, label_embedding_dim))

    @staticmethod
    def create_correlation_matrix(train_labels: torch.Tensor, num_classes: int, p: float, tau: float) -> torch.Tensor:
        co_occurrence = train_labels.T @ train_labels
        num_labels_per_class = torch.diagonal(co_occurrence)

        conditional_prob = co_occurrence / num_labels_per_class.unsqueeze(1)
        conditional_prob.nan_to_num_(0)

        adj_matrix = (conditional_prob > tau).float()

        adj_matrix_no_self_loop = adj_matrix - torch.eye(num_classes, device=adj_matrix.device)
        sum_neighbors = adj_matrix_no_self_loop.sum(dim=1)

        weights_p = p / sum_neighbors
        weights_p.nan_to_num_(0)

        reweighted_adj = adj_matrix_no_self_loop * weights_p.unsqueeze(1)
        reweighted_adj.fill_diagonal_(1 - p)

        return cast(torch.Tensor, reweighted_adj)

    def set_correlation_matrix(self, train_labels: torch.Tensor) -> None:
        corr_matrix = self.create_correlation_matrix(
            train_labels, self.num_classes, self.p_reweight, self.tau_threshold
        )
        self.correlation_matrix.copy_(corr_matrix)

    def set_label_embeddings(self, label_embeddings: torch.Tensor) -> None:
        self.label_embeddings.copy_(label_embeddings)

    def forward(self, bert_features: torch.Tensor) -> torch.Tensor:
        classifiers: torch.Tensor = self.label_embeddings
        for gcn_layer, activation in zip(self.gcn_layers, self.activations, strict=True):
            classifiers = gcn_layer(self.correlation_matrix, classifiers)
            classifiers = activation(classifiers)

        return torch.matmul(bert_features, classifiers.T)

    def dump(self, path: Path) -> None:
        metadata = GCNModelDumpMetadata(
            num_classes=self.num_classes,
            bert_feature_dim=self.bert_feature_dim,
            label_embedding_dim=self.label_embedding_dim,
            gcn_hidden_dims=self.gcn_hidden_dims,
            p_reweight=self.p_reweight,
            tau_threshold=self.tau_threshold,
        )
        with (path / self._metadata_dict_name).open("w", encoding="utf-8") as file:
            json.dump(metadata.model_dump(mode="json"), file, indent=4, ensure_ascii=False)

        device = self.device
        self.cpu()
        torch.save(self.state_dict(), path / self._state_dict_name)
        self.to(device)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> Self:
        with (path / cls._metadata_dict_name).open() as file:
            metadata = GCNModelDumpMetadata(**json.load(file))
        device = device or detect_device()

        instance = cls(**metadata.model_dump())
        state_dict = torch.load(path / cls._state_dict_name, map_location="cpu")
        instance.load_state_dict(state_dict)

        instance = instance.to(device)
        instance.eval()
        return instance
