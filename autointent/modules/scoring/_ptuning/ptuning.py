"""PTuningScorer class for ptuning-based classification."""

from pathlib import Path
from typing import Any

from peft import PromptEncoderConfig, get_peft_model

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
from autointent._dump_tools import Dumper
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

    def __init__(
        self,
        classification_model_config: HFModelConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        report_to: REPORTERS_NAMES | None = None,  # type: ignore[valid-type]
        **ptuning_kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(
            classification_model_config=classification_model_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=report_to,
        )
        self._ptuning_config = PromptEncoderConfig(task_type="SEQ_CLS", **ptuning_kwargs)

    @classmethod
    def from_context(
        cls,
        context: Context,
        classification_model_config: HFModelConfig | str | dict[str, Any] | None = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        **ptuning_kwargs: Any,  # noqa: ANN401
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
            classification_model_config = context.resolve_transformer()

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

    def _initialize_model(self) -> Any:  # noqa: ANN401
        """Initialize the model with P-tuning configuration."""
        model = super()._initialize_model()
        return get_peft_model(model, self._ptuning_config)

    def dump(self, path: str) -> None:
        Dumper.dump(self, Path(path), exclude=[PromptEncoderConfig])
