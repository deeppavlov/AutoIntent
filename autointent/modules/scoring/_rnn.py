from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.optim import Adam

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
from autointent.configs import RNNConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer


class RNNScorer(BaseScorer):
    """Scorer based on RNN model for text classification."""

    name = "rnn"
    supports_multiclass = True
    supports_multilabel = True

    def __init__(
        self,
        rnn_config: RNNConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        report_to: REPORTERS_NAMES | None = None,  # type: ignore  # noqa: PGH003
    ) -> None:
        """Initialize the RNN scorer."""
        self.rnn_config = RNNConfig.from_search_config(rnn_config)
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size or self.rnn_config.batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.report_to = report_to
        self._artifact = None
        self._device = self.rnn_config.device or ("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_context(
        cls,
        context: Context,
        rnn_config: RNNConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
    ) -> "RNNScorer":
        """Create a RNNScorer from context."""
        report_to = context.logging_config.report_to

        return cls(
            rnn_config=rnn_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=report_to,
        )

    def get_embedder_config(self) -> dict[str, Any]:
        """Get the configuration of the embedder."""
        return self.rnn_config.model_dump()

    def __initialize_model(self, vocab_size: int) -> None:
        """Initialize the RNN model."""
        self._model = SupervisedRNNClassifier(
            vocab_size=vocab_size,
            n_classes=self._n_classes,
            embed_dim=self.rnn_config.embed_dim,
            hidden_dim=self.rnn_config.hidden_dim,
            n_layers=self.rnn_config.n_layers,
            padding_idx=self.rnn_config.padding_idx,
            dropout=self.rnn_config.dropout,
            pretrained_embs=self.rnn_config.pretrained_embs,
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

        max_len = min(max(len(seq) for seq in sequences), self.rnn_config.max_seq_length)
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
                outputs, _ = self._model(batch_x)
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
                outputs, _ = self._model(batch_x)

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


class SupervisedRNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 2,
        padding_idx: int = 0,
        dropout: float = 0.1,
        pretrained_embs: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if pretrained_embs is not None:
            _, embed_dim = pretrained_embs.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embs, freeze=True)  # type: ignore[no-untyped-call]
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(text)
        outputs, (hidden, _) = self.rnn(embedded)
        return self.fc(outputs[:, -1]), hidden[-1]
