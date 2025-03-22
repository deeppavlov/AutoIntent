"""BertScorer class for transformer-based classification with LoRA."""

import tempfile
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
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


class TokenizerConfig:
    """Configuration for tokenizer parameters."""

    def __init__(
        self,
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> None:
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation


class BERTLoRAScorer(BaseScorer):
    name = "lora"
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
        tokenizer_config: TokenizerConfig | None = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ) -> None:
        self.model_config = EmbedderConfig.from_search_config(model_config)
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self._multilabel = False
        self.tokenizer_config = tokenizer_config or TokenizerConfig()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

    @classmethod
    def from_context(
        cls,
        context: Context,
        model_config: EmbedderConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        tokenizer_config: TokenizerConfig | None = None,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ) -> "BERTLoRAScorer":
        if model_config is None:
            model_config = context.resolve_embedder()
        return cls(
            model_config=model_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            tokenizer_config=tokenizer_config,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    def get_embedder_config(self) -> dict[str, Any]:
        return self.model_config.model_dump()

    def _validate_task(self, labels: ListOfLabels) -> None:
        """Validate the task and set _multilabel flag."""
        super()._validate_task(labels)
        self._multilabel = isinstance(labels[0], list)

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        if hasattr(self, "_model"):
            self.clear_cache()

        self._validate_task(labels)

        if self._multilabel:
            labels_array = np.array(labels)
            num_labels = labels_array.shape[1]
        else:
            num_labels = len(set(labels))

        model_name = self.model_config.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_rank,  # Rank of the low-rank matrices
            lora_alpha=self.lora_alpha,  # Scaling factor
            target_modules=["query", "value"],  # Target modules to apply LoRA
            lora_dropout=self.lora_dropout,  # Dropout rate for LoRA layers
            bias="none",  # Whether to add bias to LoRA layers
        )

        # Apply LoRA to the model
        self._model = get_peft_model(self._model, lora_config)

        use_cpu = hasattr(self.model_config, "device") and self.model_config.device == "cpu"

        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            return self._tokenizer(
                examples["text"],
                padding=self.tokenizer_config.padding,
                truncation=self.tokenizer_config.truncation,
                max_length=self.tokenizer_config.max_length,
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
                report_to="wandb",
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

        inputs = self._tokenizer(
            utterances, padding=True, truncation=True, max_length=self.tokenizer_config.max_length, return_tensors="pt"
        )

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
