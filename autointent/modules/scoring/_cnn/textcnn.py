"""TextCNN model for text classification."""

import json
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from autointent._utils import detect_device
from autointent._wrappers import BaseTorchModule


class TextCNNDumpMetadata(TypedDict):
    vocab_size: int
    n_classes: int
    embed_dim: int
    kernel_sizes: list[int]
    num_filters: int
    dropout: float
    padding_idx: int


class TextCNN(BaseTorchModule):
    """TextCNN model implementation."""

    _metadata_dict_name = "metadata.json"
    _state_dict_name = "state_dict.pt"

    def __init__(
        self,
        vocab_size: int = 0,
        n_classes: int = 0,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],  # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
        padding_idx: int = 0,
        pretrained_embs: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout
        self.padding_idx = padding_idx
        self.pretrained_embs = pretrained_embs

        if pretrained_embs is not None:
            _, embed_dim = pretrained_embs.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embs, freeze=True)  # type: ignore[no-untyped-call]
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=padding_idx)

        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        embedded: torch.Tensor = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        conved: list[torch.Tensor] = [F.relu(conv(embedded)).max(dim=2)[0] for conv in self.convs]
        concatenated: torch.Tensor = torch.cat(conved, dim=1)
        dropped: torch.Tensor = self.dropout(concatenated)
        return self.fc(dropped)  # type: ignore[no-any-return]

    def dump(self, path: Path) -> None:
        metadata = {
            "vocab_size": self.vocab_size,
            "n_classes": self.n_classes,
            "embed_dim": self.embed_dim,
            "kernel_sizes": self.kernel_sizes,
            "num_filters": self.num_filters,
            "dropout": self.dropout_rate,
            "padding_idx": self.padding_idx,
        }
        with (path / self._metadata_dict_name).open("w") as file:
            json.dump(metadata, file, indent=4)

        torch.save(self.state_dict(), path / self._state_dict_name)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "TextCNN":
        with (path / cls._metadata_dict_name).open() as file:
            metadata: TextCNNDumpMetadata = json.load(file)
        instance = cls(**metadata)
        state_dict = torch.load(path / cls._state_dict_name)
        instance.load_state_dict(state_dict)
        device = device or detect_device()
        instance.eval().to(device)
        return instance
