"""TextCNN model for text classification."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from pydantic import BaseModel
from torch import nn

from autointent._utils import detect_device
from autointent._wrappers import BaseTorchModuleWithVocab
from autointent.configs import VocabConfig


class TextCNNDumpMetadata(BaseModel):
    embed_dim: int
    n_classes: int
    kernel_sizes: list[int]
    num_filters: int
    dropout: float
    vocab_config: VocabConfig


class TextCNN(BaseTorchModuleWithVocab):
    """TextCNN model implementation.

    Note: always call :py:meth:`TextCNN.build_vocab(utterances)` before using this module.
    """

    _metadata_dict_name = "metadata.json"
    _state_dict_name = "state_dict.pt"

    def __init__(
        self,
        n_classes: int,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],  # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
        vocab_config: VocabConfig | None = None,
    ) -> None:
        super().__init__(embed_dim=embed_dim, vocab_config=vocab_config)

        self.n_classes = n_classes
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout

        # Initialize other layers
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=self.embed_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.vocab_config.vocab is None:
            msg = "Model not initialized. Call build_vocab() first."
            raise ValueError(msg)

        embedded: torch.Tensor = self.embedding(x)  # (B, T, H)
        embedded = embedded.permute(0, 2, 1)  # (B, H, T)

        # list of (B, H)
        # TODO ignore padded tokens
        conved: list[torch.Tensor] = [F.relu(conv(embedded)).max(dim=2)[0] for conv in self.convs]
        concatenated: torch.Tensor = torch.cat(conved, dim=1)  # (B, H)
        dropped: torch.Tensor = self.dropout(concatenated)
        return self.fc(dropped)  # type: ignore[no-any-return]

    def dump(self, path: Path) -> None:
        metadata = TextCNNDumpMetadata(
            embed_dim=self.embed_dim,
            n_classes=self.n_classes,
            kernel_sizes=self.kernel_sizes,
            num_filters=self.num_filters,
            dropout=self.dropout_rate,
            vocab_config=self.vocab_config,
        )
        with (path / self._metadata_dict_name).open("w", encoding="utf-8") as file:
            json.dump(metadata.model_dump(mode="json"), file, indent=4, ensure_ascii=False)

        # Move model to CPU before saving state dict
        device = self.device
        self.cpu()
        torch.save(self.state_dict(), path / self._state_dict_name)
        self.to(device)  # Move back to original device

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "TextCNN":
        with (path / cls._metadata_dict_name).open() as file:
            metadata = TextCNNDumpMetadata(**json.load(file))
        device = device or detect_device()

        # Create instance and load state dict on CPU
        instance = cls(**metadata.model_dump())
        state_dict = torch.load(path / cls._state_dict_name, map_location="cpu")
        instance.load_state_dict(state_dict)

        # Move to target device after loading
        instance = instance.to(device)
        instance.eval()
        return instance
