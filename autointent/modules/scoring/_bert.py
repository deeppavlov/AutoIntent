"""BertScorer class for transformer-based classification."""

import tempfile
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (  # type: ignore[attr-defined]
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    PrinterCallback,
    ProgressCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
from autointent.configs import EarlyStoppingConfig, HFModelConfig
from autointent.custom_types import ListOfLabels
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL
from autointent.modules.base import BaseScorer


class BertScorer(BaseScorer):
    """Scoring module for transformer-based classification using BERT models.

    This module uses a transformer model (like BERT) to perform intent classification.
    It supports both multiclass and multilabel classification tasks, with options for
    early stopping and various training configurations.

    Args:
        classification_model_config: Config of the transformer model (HFModelConfig, str, or dict)
        num_train_epochs: Number of training epochs (default: 3)
        batch_size: Batch size for training (default: 8)
        learning_rate: Learning rate for training (default: 5e-5)
        seed: Random seed for reproducibility (default: 0)
        report_to: Reporting tool for training logs (e.g., "wandb", "tensorboard")
        early_stopping_config: Configuration for early stopping during training

    Example:
    --------
    .. testcode::

        from autointent.modules import BertScorer

        # Initialize scorer with BERT model
        scorer = BertScorer(
            classification_model_config="bert-base-uncased",
            num_train_epochs=3,
            batch_size=8,
            learning_rate=5e-5,
            seed=42
        )

        # Training data
        utterances = ["This is great!", "I didn't like it", "Awesome product", "Poor quality"]
        labels = [1, 0, 1, 0]

        # Fit the model
        scorer.fit(utterances, labels)

        # Make predictions
        test_utterances = ["Good product", "Not worth it"]
        probabilities = scorer.predict(test_utterances)
    """

    name = "bert"
    supports_multiclass = True
    supports_multilabel = True
    _model: Any  # transformers AutoModel factory returns Any
    _tokenizer: Any  # transformers AutoTokenizer factory returns Any

    def __init__(
        self,
        classification_model_config: HFModelConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        report_to: REPORTERS_NAMES | Literal["none"] = "none",  # type: ignore  # noqa: PGH003
        early_stopping_config: EarlyStoppingConfig | dict[str, Any] | None = None,
        print_progress: bool = False,
    ) -> None:
        self.classification_model_config = HFModelConfig.from_search_config(classification_model_config)
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.report_to = report_to
        self.early_stopping_config = EarlyStoppingConfig.from_search_config(early_stopping_config)
        self.print_progress = print_progress

    @classmethod
    def from_context(
        cls,
        context: Context,
        classification_model_config: HFModelConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        early_stopping_config: EarlyStoppingConfig | dict[str, Any] | None = None,
    ) -> "BertScorer":
        if classification_model_config is None:
            classification_model_config = context.resolve_transformer()

        return cls(
            classification_model_config=classification_model_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            early_stopping_config=early_stopping_config,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {
            "classification_model_config": self.classification_model_config.model_dump(),
            "early_stopping_config": self.early_stopping_config.model_dump(),
        }

    def _initialize_model(self) -> Any:  # noqa: ANN401
        label2id = {i: i for i in range(self._n_classes)}
        id2label = {i: i for i in range(self._n_classes)}

        return AutoModelForSequenceClassification.from_pretrained(
            self.classification_model_config.model_name,
            trust_remote_code=self.classification_model_config.trust_remote_code,
            num_labels=self._n_classes,
            label2id=label2id,
            id2label=id2label,
            problem_type="multi_label_classification" if self._multilabel else "single_label_classification",
        )

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        self._validate_task(labels)

        self._tokenizer = AutoTokenizer.from_pretrained(self.classification_model_config.model_name)  # type: ignore[no-untyped-call]
        self._model = self._initialize_model()
        tokenized_dataset = self._get_tokenized_dataset(utterances, labels)
        self._train(tokenized_dataset)

        self._model.eval()

    def _train(self, tokenized_dataset: DatasetDict) -> None:
        """Perform training with Hugging Face Trainer API.

        Args:
            tokenized_dataset: output from :py:meth:`BertScorer._get_tokenized_dataset`
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                num_train_epochs=self.num_train_epochs,
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                seed=self.seed,
                save_strategy="epoch",
                save_total_limit=1,
                eval_strategy="epoch",
                logging_strategy="steps",
                logging_steps=10,
                report_to=self.report_to,
                fp16=self.classification_model_config.fp16,
                bf16=self.classification_model_config.bf16,
                use_cpu=self.classification_model_config.device == "cpu",
                metric_for_best_model=self.early_stopping_config.metric,
                load_best_model_at_end=self.early_stopping_config.metric is not None,
            )

            trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"],
                processing_class=self._tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=self._tokenizer),
                compute_metrics=self._get_compute_metrics(),
                callbacks=self._get_trainer_callbacks(),
            )
            if not self.print_progress:
                trainer.remove_callback(PrinterCallback)
                trainer.remove_callback(ProgressCallback)

            trainer.train()

    def _get_trainer_callbacks(self) -> list[TrainerCallback]:
        res: list[TrainerCallback] = []
        if self.early_stopping_config.metric is not None:
            res.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_config.patience,
                    early_stopping_threshold=self.early_stopping_config.threshold,
                )
            )
        return res

    def _get_tokenized_dataset(self, utterances: list[str], labels: ListOfLabels) -> DatasetDict:
        """Build tokenized dataset with "train" and "validation" splits."""
        train_utterances, val_utterances, train_labels, val_labels = train_test_split(
            utterances, labels, test_size=self.early_stopping_config.val_fraction
        )

        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            return self._tokenizer(  # type: ignore[no-any-return]
                examples["text"], return_tensors="pt", **self.classification_model_config.tokenizer_config.model_dump()
            )

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"text": train_utterances, "labels": train_labels}),
                "validation": Dataset.from_dict({"text": val_utterances, "labels": val_labels}),
            }
        )

        if self._multilabel:
            # hugging face uses F.binary_cross_entropy_with_logits under the hood
            # which requires target labels to be of float type
            dataset = dataset.map(
                lambda example: {"label": torch.tensor(example["labels"], dtype=torch.float)}, remove_columns="labels"
            )

        return dataset.map(tokenize_function, batched=True, batch_size=self.batch_size)

    def _get_compute_metrics(self) -> Callable[[EvalPrediction], dict[str, float]] | None:
        """Construct callable for computing metrics during transformer training.

        The result of this function is supposed to pass to :py:class:`transformers.Trainer`.
        """
        if self.early_stopping_config.metric is None:
            return None

        metric_name = self.early_stopping_config.metric
        metric_fn = (SCORING_METRICS_MULTILABEL | SCORING_METRICS_MULTICLASS)[metric_name]

        def compute_metrics(output: EvalPrediction) -> dict[str, float]:
            return {
                metric_name: metric_fn(output.label_ids.tolist(), output.predictions.tolist())  # type: ignore[union-attr]
            }

        return compute_metrics

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        if not hasattr(self, "_model") or not hasattr(self, "_tokenizer"):
            msg = "Model is not trained. Call fit() first."
            raise RuntimeError(msg)

        device = next(self._model.parameters()).device
        all_predictions = []
        for i in range(0, len(utterances), self.batch_size):
            batch = utterances[i : i + self.batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", **self.classification_model_config.tokenizer_config.model_dump()
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
            if self._multilabel:
                batch_predictions = torch.sigmoid(logits).cpu().numpy()
            else:
                batch_predictions = torch.softmax(logits, dim=1).cpu().numpy()
            all_predictions.append(batch_predictions)
        return np.vstack(all_predictions) if all_predictions else np.array([])

    def clear_cache(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer"):
            del self._tokenizer
