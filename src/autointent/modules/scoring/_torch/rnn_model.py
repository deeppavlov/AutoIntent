import json
from pathlib import Path

import torch
from pydantic import BaseModel
from torch import nn

from autointent._utils import detect_device
from autointent._wrappers import BaseTorchModuleWithVocab
from autointent.configs import VocabConfig


class TextRNNDumpMetadata(BaseModel):
    embed_dim: int
    n_classes: int
    hidden_dim: int
    n_layers: int
    dropout: float
    vocab_config: VocabConfig


class TextRNN(BaseTorchModuleWithVocab):
    _metadata_dict_name = "metadata.json"
    _state_dict_name = "state_dict.pt"

    def __init__(
        self,
        n_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        vocab_config: VocabConfig | None = None,
    ) -> None:
        super().__init__(embed_dim=embed_dim, vocab_config=vocab_config)

        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)  # (B, T, H)
        outputs, _ = self.rnn(embedded)  # (B, T, H)
        # (B, H), rightmost token's embedding
        # TODO ignore padded tokens
        return self.fc(outputs[:, -1])  # type: ignore[no-any-return]

    def dump(self, path: Path) -> None:
        metadata = TextRNNDumpMetadata(
            embed_dim=self.embed_dim,
            n_classes=self.n_classes,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
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
    def load(cls, path: Path, device: str | None = None) -> "TextRNN":
        with (path / cls._metadata_dict_name).open() as file:
            metadata = TextRNNDumpMetadata(**json.load(file))
        device = device or detect_device()

        # Create instance and load state dict on CPU
        instance = cls(**metadata.model_dump())
        state_dict = torch.load(path / cls._state_dict_name, map_location="cpu")
        instance.load_state_dict(state_dict)

        # Move to target device after loading
        instance = instance.to(device)
        instance.eval()
        return instance
