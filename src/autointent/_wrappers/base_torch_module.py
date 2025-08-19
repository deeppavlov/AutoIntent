"""Torch model for text classification."""

import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from typing_extensions import Self

from autointent.configs import VocabConfig


class BaseTorchModuleWithVocab(nn.Module, ABC):
    def __init__(
        self,
        embed_dim: int,
        vocab_config: VocabConfig | None = None,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_config = VocabConfig.from_search_config(vocab_config)

        # Vocabulary management
        self._unk_token = "<UNK>"  # noqa: S105
        self._pad_token = "<PAD>"  # noqa: S105
        self._unk_idx = 1

        if self.vocab_config.vocab is not None:
            self.set_vocab(self.vocab_config.vocab)

    def set_vocab(self, vocab: dict[str, Any]) -> None:
        """Save vocabulary into module's attributes and initialize embeddings matrix."""
        self.vocab_config.vocab = vocab
        self.embedding = nn.Embedding(
            num_embeddings=len(self.vocab_config.vocab),
            embedding_dim=self.embed_dim,
            padding_idx=self.vocab_config.padding_idx,
        )

    def build_vocab(self, utterances: list[str]) -> None:
        """Build vocabulary from training utterances."""
        if self.vocab_config.vocab is not None:
            msg = "Vocab is already built."
            raise RuntimeError(msg)

        word_counts: Counter[str] = Counter()
        for utterance in utterances:
            words = re.findall(r"\w+", utterance.lower())
            word_counts.update(words)

        # Create vocabulary with special tokens
        vocab = {self._pad_token: self.vocab_config.padding_idx, self._unk_token: self._unk_idx}

        # Convert Counter to list of (word, count) tuples sorted by frequency
        sorted_words = word_counts.most_common(self.vocab_config.max_vocab_size)
        for word, _ in sorted_words:
            if word not in vocab:
                vocab[word] = len(vocab)

        self.set_vocab(vocab)

    def text_to_indices(self, utterances: list[str]) -> list[list[int]]:
        """Convert utterances to padded sequences of word indices."""
        if self.vocab_config.vocab is None:
            msg = "Vocab is not built."
            raise RuntimeError(msg)

        sequences: list[list[int]] = []
        for utterance in utterances:
            words = re.findall(r"\w+", utterance.lower())
            # Convert words to indices, using UNK for unknown words
            seq = [self.vocab_config.vocab.get(word, self._unk_idx) for word in words]
            # Truncate if too long
            seq = seq[: self.vocab_config.max_seq_length]
            # Pad if too short
            seq = seq + [self.vocab_config.padding_idx] * (self.vocab_config.max_seq_length - len(seq))
            sequences.append(seq)
        return sequences

    @abstractmethod
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Compute sentence embeddings for given text.

        Args:
            text: torch tensor of shape (B, T), token ids

        Returns:
            embeddings of shape (B, H)
        """

    @abstractmethod
    def dump(self, path: Path) -> None:
        """Dump torch module to disk.

        This method encapsulates all the logic of dumping module's weights and
        hyperparameters required for initialization from disk and nice inference.

        Args:
            path: path in file system
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path, device: str | None = None) -> Self:
        """Load torch module from disk.

        This method loads all weights and hyperparameters required for
        initialization from disk and inference.

        Args:
            path: path in file system
            device: torch notation for CPU, CUDA, MPS, etc. By default, it is inferred automatically.
        """

    @property
    def device(self) -> torch.device:
        """Torch device object where this module resides."""
        return next(self.parameters()).device
