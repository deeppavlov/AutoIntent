"""BertScorer class for transformer-based classification."""

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
from autointent._callbacks import REPORTERS_NAMES
from autointent.configs import HFModelConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer


class BertScorer(BaseScorer):
    name = "transformer"
    supports_multiclass = True
    supports_multilabel = True
    _model: Any
    _tokenizer: Any

    def __init__(
        self,
        model_config: HFModelConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        report_to: REPORTERS_NAMES | None = None,  # type: ignore  # noqa: PGH003
    ) -> None:
        self.model_config = HFModelConfig.from_search_config(model_config)
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.report_to = report_to

    @classmethod
    def from_context(
        cls,
        context: Context,
        model_config: HFModelConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
    ) -> "BertScorer":
        if model_config is None:
            model_config = context.resolve_embedder()

        report_to = context.logging_config.report_to

        return cls(
            model_config=model_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=report_to,
        )

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

        model_name = self.model_config.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self._n_classes)

        use_cpu = self.model_config.device == "cpu"

        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            return self._tokenizer(  # type: ignore[no-any-return]
                examples["text"], return_tensors="pt", **self.model_config.tokenizer_config.model_dump()
            )

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
                logging_strategy="steps",
                logging_steps=10,
                report_to=self.report_to,
                use_cpu=use_cpu,
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

        all_predictions = []
        for i in range(0, len(utterances), self.batch_size):
            batch = utterances[i : i + self.batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt", **self.model_config.tokenizer_config.model_dump())
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
            if self._multilabel:
                batch_predictions = torch.sigmoid(logits).numpy()
            else:
                batch_predictions = torch.softmax(logits, dim=1).numpy()
            all_predictions.append(batch_predictions)
        return np.vstack(all_predictions) if all_predictions else np.array([])

    def clear_cache(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer"):
            del self._tokenizer
