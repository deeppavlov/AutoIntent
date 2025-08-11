"""BertScorer class for transformer-based classification with LoRA."""

from pathlib import Path
from typing import Any, Literal

from peft import LoraConfig, get_peft_model

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
from autointent._dump_tools import Dumper
from autointent.configs import EarlyStoppingConfig, HFModelConfig
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

    """

    name = "lora"

    def __init__(
        self,
        classification_model_config: HFModelConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        report_to: REPORTERS_NAMES | Literal["none"] = "none",  # type: ignore  # noqa: PGH003
        print_progress: bool = False,
        **lora_kwargs: Any,  # noqa: ANN401
    ) -> None:
        # early stopping doesnt work with lora for now https://github.com/huggingface/transformers/issues/38130
        early_stopping_config = EarlyStoppingConfig(metric=None)  # disable early stopping

        super().__init__(
            classification_model_config=classification_model_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=report_to,
            early_stopping_config=early_stopping_config,
            print_progress=print_progress,
        )
        self._lora_config = LoraConfig(**lora_kwargs)

    @classmethod
    def from_context(
        cls,
        context: Context,
        classification_model_config: HFModelConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        **lora_kwargs: Any,  # noqa: ANN401
    ) -> "BERTLoRAScorer":
        if classification_model_config is None:
            classification_model_config = context.resolve_transformer()
        return cls(
            classification_model_config=classification_model_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=context.logging_config.report_to,
            **lora_kwargs,
        )

    def _initialize_model(self) -> Any:  # noqa: ANN401
        model = super()._initialize_model()
        return get_peft_model(model, self._lora_config)

    def dump(self, path: str) -> None:
        Dumper.dump(self, Path(path), exclude=[LoraConfig])
