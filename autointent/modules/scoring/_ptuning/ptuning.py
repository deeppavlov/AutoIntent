"""PTuningScorer class for ptuning-based classification."""

from typing import Any

import torch
from peft import PromptEncoderConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
)

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
from autointent.configs import HFModelConfig
from autointent.modules.scoring._bert import BertScorer


class PTuningScorer(BertScorer):
    """PEFT P-tuning scorer.

    Args:
        classification_model_config: Config of the base transformer model (HFModelConfig, str, or dict)
        num_train_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        seed: Random seed for reproducibility
        report_to: Reporting tool for training logs
        **ptuning_kwargs: Arguments for `PromptEncoderConfig <https://huggingface.co/docs/peft/package_reference/p_tuning#peft.PromptEncoderConfig>`_

    Example:
    --------
    .. testcode::

        from autointent.modules import PTuningScorer
        scorer = PTuningScorer(
            classification_model_config="prajjwal1/bert-tiny",
            num_train_epochs=3,
            batch_size=8,
            task_type="SEQ_CLS",
            num_virtual_tokens=10,
            seed=42
        )
        utterances = ["hello", "goodbye", "allo", "sayonara"]
        labels = [0, 1, 0, 1]
        scorer.fit(utterances, labels)
        test_utterances = ["hi", "bye"]
        probabilities = scorer.predict(test_utterances)
        print(probabilities)

    .. testoutput::

        [[0.49925193 0.50074804]
        [0.4944601  0.5055399 ]]

    """

    name = "ptuning"
    supports_multiclass = True
    supports_multilabel = True
    _model: Any
    _tokenizer: Any

    def __init__(
        self,
        classification_model_config: HFModelConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        report_to: REPORTERS_NAMES | None = None,  # type: ignore[valid-type]
        **ptuning_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            classification_model_config=classification_model_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=report_to,
        )
        self._ptuning_config = PromptEncoderConfig(**ptuning_kwargs)  # type: ignore[arg-type]
        torch.manual_seed(seed)

    @classmethod
    def from_context(
        cls,
        context: Context,
        classification_model_config: HFModelConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        **ptuning_kwargs: dict[str, Any],
    ) -> "PTuningScorer":
        """Create a PTuningScorer instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            classification_model_config: Config of the base model, or None to use the best embedder
            num_train_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            seed: Random seed for reproducibility
            **ptuning_kwargs: Arguments for PromptEncoderConfig
        """
        if classification_model_config is None:
            classification_model_config = context.resolve_embedder()

        report_to = context.logging_config.report_to

        return cls(
            classification_model_config=classification_model_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=report_to,
            **ptuning_kwargs,
        )

    def _initialize_model(self) -> None:
        """Initialize the model with P-tuning configuration."""
        model_name = self.classification_model_config.model_name
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self._n_classes,
            problem_type="multi_label_classification" if self._multilabel else "single_label_classification",
            trust_remote_code=self.classification_model_config.trust_remote_code,
            return_dict=True,
        )
        self._model = get_peft_model(self._model, self._ptuning_config)
