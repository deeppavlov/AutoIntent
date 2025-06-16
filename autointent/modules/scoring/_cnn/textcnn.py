"""TextCNN model for text classification."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from autointent._utils import detect_device
from autointent._wrappers import BaseTorchModule


class TextCNNDumpMetadata(TypedDict):
    n_classes: int
    embed_dim: int
    kernel_sizes: list[int]
    num_filters: int
    dropout: float
    padding_idx: int
    vocab: dict[str, int]
    max_seq_length: int
    max_vocab_size: int | None


class TextCNN(BaseTorchModule):
    """TextCNN model implementation."""

    _metadata_dict_name = "metadata.json"
    _state_dict_name = "state_dict.pt"

    def __init__(
        self,
        n_classes: int,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],  # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
        padding_idx: int = 0,
        max_seq_length: int = 50,
        vocab: dict[str, int] | None = None,
        max_vocab_size: int | None = None,
    ) -> None:
        super().__init__()

        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout
        self.padding_idx = padding_idx
        self.max_seq_length = max_seq_length

        # Vocabulary management
        self._unk_token = "<UNK>"  # noqa: S105
        self._pad_token = "<PAD>"  # noqa: S105
        self._unk_idx = 1
        self._pad_idx = padding_idx
        self.max_vocab_size = max_vocab_size

        if vocab is not None:
            self._vocab = vocab
            self.embedding = nn.Embedding(
                num_embeddings=len(self._vocab), embedding_dim=self.embed_dim, padding_idx=self.padding_idx
            )

        # Initialize other layers
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), n_classes)

    def build_vocab(self, utterances: list[str]) -> None:
        """Build vocabulary from training utterances."""
        if hasattr(self, "_vocab"):
            msg = "Vocab is already built."
            raise RuntimeError(msg)

        word_counts: Counter[str] = Counter()
        for utterance in utterances:
            words = re.findall(r"\w+", utterance.lower())
            word_counts.update(words)

        # Create vocabulary with special tokens
        self._vocab = {self._pad_token: self._pad_idx, self._unk_token: self._unk_idx}

        # Convert Counter to list of (word, count) tuples sorted by frequency
        sorted_words = word_counts.most_common(self.max_vocab_size)
        for word, _ in sorted_words:
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab)

        # Update vocab_size and initialize embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=len(self._vocab), embedding_dim=self.embed_dim, padding_idx=self.padding_idx
        )

    def text_to_indices(self, utterances: list[str]) -> list[list[int]]:
        """Convert utterances to padded sequences of word indices."""
        sequences: list[list[int]] = []
        for utterance in utterances:
            words = re.findall(r"\w+", utterance.lower())
            # Convert words to indices, using UNK for unknown words
            seq = [self._vocab.get(word, self._unk_idx) for word in words]
            # Truncate if too long
            seq = seq[: self.max_seq_length]
            # Pad if too short
            seq = seq + [self._pad_idx] * (self.max_seq_length - len(seq))
            sequences.append(seq)
        return sequences

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        if not hasattr(self, "_vocab"):
            msg = "Model not initialized. Call build_vocab() first."
            raise ValueError(msg)

        embedded: torch.Tensor = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        conved: list[torch.Tensor] = [F.relu(conv(embedded)).max(dim=2)[0] for conv in self.convs]
        concatenated: torch.Tensor = torch.cat(conved, dim=1)
        dropped: torch.Tensor = self.dropout(concatenated)
        return self.fc(dropped)  # type: ignore[no-any-return]

    def dump(self, path: Path) -> None:
        metadata: TextCNNDumpMetadata = {
            "n_classes": self.n_classes,
            "embed_dim": self.embed_dim,
            "kernel_sizes": self.kernel_sizes,
            "num_filters": self.num_filters,
            "dropout": self.dropout_rate,
            "padding_idx": self.padding_idx,
            "vocab": self._vocab,
            "max_seq_length": self.max_seq_length,
            "max_vocab_size": self.max_vocab_size,
        }
        with (path / self._metadata_dict_name).open("w") as file:
            json.dump(metadata, file, indent=4)

        # Move model to CPU before saving state dict
        device = self.device
        self.cpu()
        torch.save(self.state_dict(), path / self._state_dict_name)
        self.to(device)  # Move back to original device

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "TextCNN":
        with (path / cls._metadata_dict_name).open() as file:
            metadata: TextCNNDumpMetadata = json.load(file)
        device = device or detect_device()

        # Create instance and load state dict on CPU
        instance = cls(**metadata)
        state_dict = torch.load(path / cls._state_dict_name, map_location="cpu")
        instance.load_state_dict(state_dict)

        # Move to target device after loading
        instance = instance.to(device)
        instance.eval()
        return instance
