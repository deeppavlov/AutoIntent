import logging
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autointent._wrappers import BaseTorchModuleWithVocab
from autointent.configs import EarlyStoppingConfig, TorchTrainingConfig, VocabConfig
from autointent.custom_types import ListOfLabels
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL, ScoringMetricFn
from autointent.modules.base import BaseScorer

logger = logging.getLogger(__name__)


class BaseTorchScorer(BaseScorer):
    supports_multiclass = True
    supports_multilabel = True

    _best_model_weights = "best_model.pt"

    def __init__(
        self,
        torch_config: TorchTrainingConfig | dict[str, Any] | None = None,
        vocab_config: VocabConfig | dict[str, Any] | None = None,
        early_stopping_config: EarlyStoppingConfig | dict[str, Any] | None = None,
    ) -> None:
        self.torch_config = TorchTrainingConfig.from_search_config(torch_config)
        self.vocab_config = VocabConfig.from_search_config(vocab_config)
        self.early_stopping_config = EarlyStoppingConfig.from_search_config(early_stopping_config)

    @abstractmethod
    def _init_model(self) -> BaseTorchModuleWithVocab: ...

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        self._validate_task(labels)

        self._model = self._init_model()
        logger.debug("Torch model initialized: %s", type(self._model).__name__)

        self._model.build_vocab(utterances)
        logger.debug("Vocabulary built for model.")
        x = self._model.text_to_indices(utterances)
        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(labels, dtype=torch.long if not self._multilabel else torch.float)

        # Split data for early stopping if configured
        if self.early_stopping_config.metric is not None:
            logger.debug(
                "Splitting data for early stopping with val_fraction=%.2f", self.early_stopping_config.val_fraction
            )
            train_x, val_x, train_y, val_y = train_test_split(
                x_tensor,
                y_tensor,
                test_size=self.early_stopping_config.val_fraction,
                random_state=self.torch_config.seed,
            )
        else:
            logger.debug("No early stopping metric configured; using all data for training.")
            train_x, val_x, train_y, val_y = x_tensor, None, y_tensor, None

        self._train_model(train_x, train_y, val_x, val_y)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        if not hasattr(self, "_model"):
            msg = "Scorer is not trained. Call fit() first."
            logger.error(msg)
            raise RuntimeError(msg)

        logger.debug("Predicting for %d utterances.", len(utterances))
        x = self._model.text_to_indices(utterances)
        x_tensor = torch.tensor(x, dtype=torch.long)

        return self._predict_tensors(x_tensor)

    def clear_cache(self) -> None:
        self.vocab_config.vocab = None

        if hasattr(self, "_model"):
            logger.debug("Clearing model and CUDA cache.")
            self._model.vocab_config.vocab = None
            del self._model
            torch.cuda.empty_cache()

    def _train_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor | None = None,
        val_y: torch.Tensor | None = None,
    ) -> None:
        if not hasattr(self, "_model"):
            msg = "Scorer is not initialized"
            logger.error(msg)
            raise ValueError(msg)

        logger.debug("Preparing dataloader and optimizer for training.")
        train_dataset = TensorDataset(train_x, train_y)
        train_dataloader = DataLoader(train_dataset, batch_size=self.torch_config.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss() if not self._multilabel else nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.torch_config.learning_rate)

        self._model.to(self.torch_config.device)
        logger.debug("Model moved to device: %s", self.torch_config.device)

        # Early stopping variables
        best_metric = float("-inf")
        patience_counter = 0
        best_model_path: Path | None = None

        # Get metric function if early stopping is enabled
        metric_fn = None
        if self.early_stopping_config.metric is not None:
            metrics_dict = SCORING_METRICS_MULTILABEL if self._multilabel else SCORING_METRICS_MULTICLASS
            metric_fn = metrics_dict[self.early_stopping_config.metric]
            logger.debug("Using early stopping with metric: %s", self.early_stopping_config.metric)

        with tempfile.TemporaryDirectory() as temp_dir:
            for epoch in range(self.torch_config.num_train_epochs):
                logger.debug("Epoch %d/%d: Training phase started.", epoch + 1, self.torch_config.num_train_epochs)
                # Training phase
                self._model.train()
                for batch_x, batch_y in train_dataloader:
                    optimizer.zero_grad()
                    outputs = self._model(batch_x.to(self.torch_config.device))
                    loss = criterion(outputs, batch_y.to(self.torch_config.device))
                    loss.backward()
                    optimizer.step()

                # Validation phase for early stopping
                if val_x is not None and val_y is not None and metric_fn is not None:
                    current_metric = self._validate_epoch(val_x, val_y, metric_fn)
                    logger.debug(
                        "Epoch %d: Validation metric (%s): %.4f",
                        epoch + 1,
                        self.early_stopping_config.metric,
                        current_metric,
                    )

                    # Check early stopping
                    should_stop, best_metric, patience_counter, best_model_path = self._should_stop_early(
                        current_metric, best_metric, patience_counter, best_model_path, temp_dir
                    )

                    if should_stop:
                        logger.debug("Early stopping triggered at epoch %d.", epoch + 1)
                        break

            # this is triggered only if early stoppping is enabled
            if best_model_path is not None:
                logger.debug("Loading best model weights from: %s", best_model_path)
                self._model.load_state_dict(torch.load(best_model_path))

        self._model.eval()
        logger.debug("Training complete. Model set to eval mode.")

    def _should_stop_early(
        self,
        current_metric: float,
        best_metric: float,
        patience_counter: int,
        best_model_path: Path | None,
        temp_dir: str,
    ) -> tuple[bool, float, int, Path | None]:
        """Check if training should stop early based on the current metric.

        Returns:
            Tuple of (should_stop, best_metric, patience_counter, best_model_path)
        """
        logger.debug(
            "Early stopping check: current_metric=%.4f, best_metric=%.4f, patience_counter=%d",
            current_metric,
            best_metric,
            patience_counter,
        )
        # Check if metric improved
        if current_metric > best_metric + self.early_stopping_config.threshold:
            best_metric = current_metric
            patience_counter = 0

            # Save best model
            if best_model_path is not None:
                # Remove previous best model
                best_model_path.unlink()

            best_model_path = Path(temp_dir) / self._best_model_weights
            torch.save(self._model.state_dict(), best_model_path)
            logger.debug("New best model saved at: %s", best_model_path)
        else:
            patience_counter += 1
            logger.debug("No improvement. Patience counter increased to %d.", patience_counter)

        # Early stopping check
        should_stop = patience_counter > self.early_stopping_config.patience
        if should_stop:
            logger.debug("Patience exceeded. Should stop training.")

        return should_stop, best_metric, patience_counter, best_model_path

    def _validate_epoch(self, val_x: torch.Tensor, val_y: torch.Tensor, metric_fn: ScoringMetricFn) -> float:
        """Validate the model for one epoch and return the metric value."""
        self._model.eval()
        logger.debug("Validating epoch on %d samples.", len(val_x))

        val_predictions = self._predict_tensors(val_x)
        val_labels: ListOfLabels = val_y.cpu().numpy().tolist()

        # Calculate metric
        metric_value = metric_fn(val_labels, val_predictions)
        logger.debug("Validation metric value: %.4f", metric_value)
        return metric_value

    def _predict_tensors(self, x_tensor: torch.Tensor) -> npt.NDArray[Any]:
        """Predict probabilities for tensor inputs."""
        self._model.eval()
        logger.debug("Predicting tensors for batch size: %d", self.torch_config.batch_size)
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

        logger.debug("Prediction complete. Returning concatenated results.")
        return np.concatenate(all_probs, axis=0)

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {}
