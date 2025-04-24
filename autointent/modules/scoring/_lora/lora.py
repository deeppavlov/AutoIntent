"""BertScorer class for transformer-based classification with LoRA."""

from typing import Any

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
from autointent.configs import HFModelConfig
from autointent.modules.scoring._bert import BertScorer


class BERTLoRAScorer(BertScorer):
    """BERTLoRAScorer class for transformer-based classification with LoRA (Low-Rank Adaptation).

    Args:
        classification_model_config: Config of the base transformer model (HFModelConfig, str, or dict)
        num_train_epochs: Number of training epochs (default: 3)
        batch_size: Batch size for training (default: 8)
        learning_rate: Learning rate for training (default: 5e-5)
        seed: Random seed for reproducibility (default: 0)
        report_to: Reporting tool for training logs
        **lora_kwargs: Arguments for `LoraConfig <https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig>`_

    Example:
    --------
    .. testcode::

        from autointent.modules import BERTLoRAScorer

        # Initialize scorer with LoRA configuration
        scorer = BERTLoRAScorer(
            classification_model_config="bert-base-uncased",
            num_train_epochs=3,
            batch_size=8,
            learning_rate=5e-5,
            seed=42,
            r=8,  # LoRA rank
            lora_alpha=16,  # LoRA alpha
        )

        # Training data
        utterances = ["This is great!", "I didn't like it", "Awesome product", "Poor quality"]
        labels = [1, 0, 1, 0]  # Binary classification

        # Fit the model
        scorer.fit(utterances, labels)

        # Make predictions
        test_utterances = ["Good product", "Not worth it"]
        probabilities = scorer.predict(test_utterances)
        print(probabilities)

    .. testoutput::

        [[0.89 0.11]
        [0.23 0.77]]
    """

    name = "lora"
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
        **lora_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            classification_model_config=classification_model_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=report_to,
            )
        self._lora_config = LoraConfig(**lora_kwargs) # type: ignore[arg-type]

    @classmethod
    def from_context(
        cls,
        context: Context,
        classification_model_config: HFModelConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        **lora_kwargs: dict[str, Any],
    ) -> "BERTLoRAScorer":
        if classification_model_config is None:
            classification_model_config = context.resolve_embedder()
        return cls(
            classification_model_config=classification_model_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=context.logging_config.report_to,
            **lora_kwargs,
        )

    def __initialize_model(self) -> None:
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.classification_model_config.model_name,
            num_labels=self._n_classes,
            problem_type="multi_label_classification" if self._multilabel else "single_label_classification",
            trust_remote_code=self.classification_model_config.trust_remote_code,
            )
        self._model = get_peft_model(self._model, self._lora_config)
