"""TransformerScorer class for transformer-based classification."""

import tempfile
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from autointent import Context
from autointent.configs import EmbedderConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer


class TransformerScorer(BaseScorer):
    name = "transformer"
    supports_multiclass = True
    supports_multilabel = True
    _multilabel: bool
    _model: Any
    _tokenizer: Any

    def __init__(
        self,
        model_config: EmbedderConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
    ) -> None:
        self.model_config = EmbedderConfig.from_search_config(model_config)
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    @classmethod
    def from_context(
        cls,
        context: Context,
        model_config: EmbedderConfig | str | None = None,
    ) -> "TransformerScorer":
        if model_config is None:
            model_config = context.resolve_embedder()
        return cls(model_config=model_config)

    def get_embedder_config(self) -> dict[str, Any]:
        return self.model_config.model_dump()

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        if hasattr(self, "_model"):
            self.clear_cache()

        self._validate_task(labels)

        if self._multilabel:
            labels_array = np.array(labels) if not isinstance(labels, np.ndarray) else labels
            num_labels = labels_array.shape[1]
        else:
            num_labels = len(set(labels))

        model_name = self.model_config.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            return self._tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        dataset = Dataset.from_dict({"text": utterances, "labels": labels})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                num_train_epochs=self.num_train_epochs,
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                seed=self.seed,
                save_strategy="no",
                logging_strategy="no",
                report_to="none",
            )

            trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self._tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=self._tokenizer),
            )

            trainer.train()

        self._model.eval()

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        if not hasattr(self, "_model") or not hasattr(self, "_tokenizer"):
            msg = "Model is not trained. Call fit() first."
            raise RuntimeError(msg)

        inputs = self._tokenizer(utterances, padding=True, truncation=True, max_length=128, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        if self._multilabel:
            return torch.sigmoid(logits).numpy()
        return torch.softmax(logits, dim=1).numpy()


    def clear_cache(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer"):
            del self._tokenizer
