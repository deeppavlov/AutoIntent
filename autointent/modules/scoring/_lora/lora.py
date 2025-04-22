"""BertScorer class for transformer-based classification with LoRA."""

from typing import Any

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
from autointent.configs import HFModelConfig
from autointent.modules.scoring._bert import BertScorer


class BERTLoRAScorer(BertScorer):
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
        self._lora_config = LoraConfig(**lora_kwargs) # type: ignore[valid-type]

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
