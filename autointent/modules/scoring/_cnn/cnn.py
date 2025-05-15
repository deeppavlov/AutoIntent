from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.optim import Adam

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
from autointent.configs import CNNConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer
from autointent.modules.scoring._cnn.textcnn import TextCNN


class CNNScorer(BaseScorer):
    """Scorer based on CNN model for text classification."""

    name = "cnn"
    supports_multiclass = True
    supports_multilabel = True

    def __init__(
        self,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],
        num_filters: int = 100,
        dropout: float = 0.1,
        cnn_config: CNNConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        report_to: REPORTERS_NAMES | None = None,  # type: ignore  # noqa: PGH003
    ) -> None:
        """Initialize the CNN scorer."""
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout = dropout
        self.cnn_config = CNNConfig.from_search_config(cnn_config)
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size or self.cnn_config.batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.report_to = report_to
        self._artifact = None
        self._device = self.cnn_config.device or ("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_context(
        cls,
        context: Context,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],
        num_filters: int = 100,
        dropout: float = 0.1,
        cnn_config: CNNConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
    ) -> "CNNScorer":
        """Create a CNNScorer from context."""
        report_to = context.logging_config.report_to

        return cls(
            embed_dim=embed_dim,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            dropout=dropout,
            cnn_config=cnn_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=report_to,
        )

    def get_embedder_config(self) -> dict[str, Any]:
        """Get the configuration of the embedder."""
        config = self.cnn_config.model_dump()
        config.update({
            "embed_dim": self.embed_dim,
            "kernel_sizes": self.kernel_sizes,
            "num_filters": self.num_filters,
            "dropout": self.dropout,
        })
        return config

    def __initialize_model(self, vocab_size: int) -> None:
        """Initialize the CNN model."""
        self._model = TextCNN(
            vocab_size=vocab_size,
            n_classes=self._n_classes,
            embed_dim=self.embed_dim,
            kernel_sizes=self.kernel_sizes,
            num_filters=self.num_filters,
            dropout=self.dropout,
            padding_idx=self.cnn_config.padding_idx,
            pretrained_embs=None,
        )
        self._model.to(self.device)

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        """Fit the model to the given data."""
        if hasattr(self, "_model"):
            self.clear_cache()
        self._validate_task(labels)
        self._create_vocab(utterances)
        self.__initialize_model(len(self._vocab))
        x = self._texts_to_sequences(utterances)
        y = torch.tensor(labels, dtype=torch.float) if self._multilabel else torch.tensor(labels, dtype=torch.long)
        self._train_model(x, y)

    def _create_vocab(self, utterances: list[str]) -> None:
        """Create vocabulary from utterances."""
        unique_words = set()
        for text in utterances:
            for word in text.lower().split():
                unique_words.add(word)

        self._vocab = {"<PAD>": 0, "<UNK>": 1}
        for i, word in enumerate(unique_words):
            self._vocab[word] = i + 2

    def _texts_to_sequences(self, texts: list[str]) -> torch.Tensor:
        """Convert texts to sequences using the vocabulary."""
        sequences = [[self._vocab.get(word, self._vocab["<UNK>"]) for word in text.lower().split()] for text in texts]

        max_len = min(max(len(seq) for seq in sequences), self.cnn_config.max_seq_length)
        padded_sequences = [
            seq[:max_len] if len(seq) > max_len else seq + [self._vocab["<PAD>"]] * (max_len - len(seq))
            for seq in sequences
        ]

        return torch.tensor(padded_sequences, dtype=torch.long)

    def _train_model(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Train the model."""
        self._model.train()
        optimizer = Adam(self._model.parameters(), lr=self.learning_rate)

        criterion = nn.BCEWithLogitsLoss() if self._multilabel else nn.CrossEntropyLoss()

        x = x.to(self._device)
        y = y.to(self._device)

        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        torch.manual_seed(self.seed)

        for _epoch in range(self.num_train_epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self._model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self._model.eval()

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """Predict probabilities for utterances."""
        if not hasattr(self, "_model") or not hasattr(self, "_vocab"):
            msg = "Model is not trained. Call fit() first."
            raise RuntimeError(msg)

        x = self._texts_to_sequences(utterances)
        x = x.to(self.device)

        self._model.eval()
        all_predictions = []

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                batch_x = x[i : i + self.batch_size]
                outputs = self._model(batch_x)

                if self._multilabel:
                    batch_predictions = torch.sigmoid(outputs).cpu().numpy()
                else:
                    batch_predictions = torch.softmax(outputs, dim=1).cpu().numpy()

                all_predictions.append(batch_predictions)

        return np.vstack(all_predictions) if all_predictions else np.array([])

    def clear_cache(self) -> None:
        """Clear model cache."""
        if hasattr(self, "_model"):
            del self._model

    @property
    def device(self) -> str:
        """Get device used for model computations."""
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        """Set device for model computations."""
        self._device = value

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        """Return default params used in ``__init__`` method."""
        return {"cnn_config": self.cnn_config.model_dump()}
