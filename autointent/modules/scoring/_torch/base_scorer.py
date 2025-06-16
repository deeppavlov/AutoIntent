from abc import abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autointent._wrappers import BaseTorchModuleWithVocab
from autointent.configs import TorchTrainingConfig, VocabConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer


class BaseTorchScorer(BaseScorer):
    supports_multiclass = True
    supports_multilabel = True

    def __init__(
        self,
        torch_config: TorchTrainingConfig | dict[str, Any] | None = None,
        vocab_config: VocabConfig | dict[str, Any] | None = None,
    ) -> None:
        self.torch_config = TorchTrainingConfig.from_search_config(torch_config)
        self.vocab_config = VocabConfig.from_search_config(vocab_config)

    @abstractmethod
    def _init_model(self) -> BaseTorchModuleWithVocab: ...

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        self._validate_task(labels)

        self._model = self._init_model()

        self._model.build_vocab(utterances)
        x = self._model.text_to_indices(utterances)
        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(labels, dtype=torch.long if not self._multilabel else torch.float)

        self._train_model(x_tensor, y_tensor)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        if not hasattr(self, "_model"):
            msg = "Scorer is not trained. Call fit() first."
            raise RuntimeError(msg)

        x = self._model.text_to_indices(utterances)
        x_tensor = torch.tensor(x, dtype=torch.long)

        self._model.eval()
        all_probs: list[npt.NDArray[Any]] = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.torch_config.batch_size):
                batch_x = x_tensor[i : i + self.torch_config.batch_size].to(self.torch_config.device)
                outputs = self._model(batch_x)
                if self._multilabel:
                    probs = torch.sigmoid(outputs).cpu().numpy()
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0) if all_probs else np.array([])

    def clear_cache(self) -> None:
        if hasattr(self, "_model"):
            self._model.vocab_config.vocab = None
            del self._model
            torch.cuda.empty_cache()

    def _train_model(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if not hasattr(self, "_model"):
            msg = "Scorer is not initialized"
            raise ValueError(msg)

        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=self.torch_config.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss() if not self._multilabel else nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.torch_config.learning_rate)

        self._model.to(self.torch_config.device)
        self._model.train()
        for _ in range(self.torch_config.num_train_epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self._model(batch_x.to(self.torch_config.device))
                loss = criterion(outputs, batch_y.to(self.torch_config.device))
                loss.backward()
                optimizer.step()

        self._model.eval()

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {}
