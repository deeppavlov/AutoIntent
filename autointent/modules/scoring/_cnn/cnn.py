"""CNNScorer class for scoring."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
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
        max_seq_length: int = 50,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        report_to: REPORTERS_NAMES | None = None,  # type: ignore[valid-type]
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],  # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
    ) -> None:
        self.max_seq_length = max_seq_length
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.report_to = report_to
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout = dropout

    @classmethod
    def from_context(
        cls,
        context: Context,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],  # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
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
        )

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        self._validate_task(labels)

        # Initialize model
        self._model = TextCNN(
            n_classes=self._n_classes,
            embed_dim=self.embed_dim,
            kernel_sizes=self.kernel_sizes,
            num_filters=self.num_filters,
            dropout=self.dropout,
            max_seq_length=self.max_seq_length,
        )

        # Build vocabulary and convert text to indices
        self._model.build_vocab(utterances)
        x = self._model.text_to_indices(utterances)
        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(labels, dtype=torch.long if not self._multilabel else torch.float)

        # Training
        self._train_model(x_tensor, y_tensor)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        if not hasattr(self, "_model"):
            msg = "CNNScorer is not trained. Call fit() first."
            raise ValueError(msg)

        x = self._model.text_to_indices(utterances)
        x_tensor = torch.tensor(x, dtype=torch.long, device=self._model.device)

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

    def clear_cache(self) -> None:
        if hasattr(self, "_model"):
            del self._model
            torch.cuda.empty_cache()

    def _train_model(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if not hasattr(self, "_model"):
            msg = "CNNSCorer is not initialized"
            raise ValueError(msg)

        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss() if not self._multilabel else nn.BCEWithLogitsLoss()
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

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {}
