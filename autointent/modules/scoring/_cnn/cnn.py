"""CNNScorer class for scoring."""

import re
from collections import Counter
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
from autointent.configs import CNNConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer
from autointent.modules.scoring._cnn.textcnn import TextCNN


class CNNScorer(BaseScorer):
    """Convolutional Neural Network (CNN) scorer for intent classification."""

    name = "cnn"
    supports_multilabel = True
    supports_multiclass = True

    def __init__(
        self,
        num_train_epochs: int = 3,
        learning_rate: float = 5e-5,
        seed: int = 0,
        report_to: REPORTERS_NAMES | None = None,  # type: ignore[valid-type]
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5], # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
        batch_size: int = 8,
        cnn_config: CNNConfig | str | dict[str, Any] | None = None,
    ) -> None:
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.report_to = report_to
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout = dropout
        self.cnn_config = CNNConfig.from_search_config(cnn_config)

        # Will be initialized during fit()
        self._model: TextCNN | None = None
        self._vocab: dict[str, int] | None = None
        self._unk_token = "<UNK>"  # noqa: S105
        self._pad_token = "<PAD>"  # noqa: S105
        self._n_classes: int = 0
        self._multilabel: bool = False
        self._pad_idx = self.cnn_config.padding_idx
        self._unk_idx = self.cnn_config.unknown_idx
        self.batch_size = batch_size
        self.max_seq_length = self.cnn_config.max_seq_length

    @classmethod
    def from_context(
        cls,
        context: Context,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5], # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
        cnn_config: CNNConfig | str | dict[str, Any] | None = None
    ) -> "CNNScorer":
        return cls(
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=context.logging_config.report_to,
            embed_dim=embed_dim,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            dropout=dropout,
            cnn_config=cnn_config
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {"cnn_config": self.cnn_config.model_dump()}

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        self._validate_task(labels)
        self._multilabel = isinstance(labels[0], (list, np.ndarray)) # noqa: UP038

        # Build vocabulary and tokenize
        self._build_vocab(utterances)

        # Convert text to padded indices
        x = self._text_to_indices(utterances)
        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(
            labels, dtype=torch.long if not self._multilabel else torch.float
        )

        # Initialize model
        if self._vocab is None:
            msg = "Vocabulary not built"
            raise ValueError(msg)

        self._model = TextCNN(
            vocab_size=len(self._vocab),
            n_classes=self._n_classes,
            embed_dim=self.embed_dim,
            kernel_sizes=self.kernel_sizes,
            num_filters=self.num_filters,
            dropout=self.dropout,
            padding_idx=self._pad_idx
        )

        # Training
        self._train_model(x_tensor, y_tensor)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        if self._model is None:
            msg = "Model not trained. Call fit() first."
            raise ValueError(msg)

        x = self._text_to_indices(utterances)
        x_tensor = torch.tensor(x, dtype=torch.long)

        self._model.eval()
        all_probs: list[npt.NDArray[Any]] = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch_x = x_tensor[i : i + self.batch_size]
                outputs = self._model(batch_x)
                if self._multilabel:
                    probs = torch.sigmoid(outputs).cpu().numpy()
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0) if all_probs else np.array([])

    def _build_vocab(self, utterances: list[str]) -> None:
        """Build vocabulary from training utterances."""
        word_counts: Counter[str] = Counter()
        for utterance in utterances:
            words = re.findall(r"\w+", utterance.lower())
            word_counts.update(words)

        # Create vocabulary with special tokens
        self._vocab = {self._pad_token: self._pad_idx, self._unk_token: self._unk_idx}

        # Convert Counter to list of (word, count) tuples sorted by frequency
        sorted_words = word_counts.most_common()
        for word, _ in sorted_words:
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab)

    def _text_to_indices(self, utterances: list[str]) -> list[list[int]]:
        """Convert utterances to padded sequences of word indices."""
        if self._vocab is None:
            msg = "Vocabulary not built"
            raise ValueError(msg)

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

    def clear_cache(self) -> None:
        self._model = None
        torch.cuda.empty_cache()

    def _train_model(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if self._model is None:
            msg = "Model not initialized"
            raise ValueError(msg)

        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = (
            nn.CrossEntropyLoss() if not self._multilabel else nn.BCEWithLogitsLoss()
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        self._model.train()
        for _ in range(self.num_train_epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self._model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        self._model.eval()
