from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from typing_extensions import assert_never

from autointent.custom_types import FloatFromZeroToOne
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL

if TYPE_CHECKING:
    from typing_extensions import Self


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    padding: bool | Literal["longest", "max_length", "do_not_pad"] = True
    truncation: bool = True
    max_length: PositiveInt | None = Field(None, description="Maximum length of input sequences.")


class EmbedderFineTuningConfig(BaseModel):
    epoch_num: int
    batch_size: int
    margin: float = Field(default=0.5)
    learning_rate: float = Field(default=2e-5)
    warmup_ratio: float = Field(default=0.1)
    early_stopping_patience: int = Field(default=1)
    early_stopping_threshold: float = Field(default=0.0)
    val_fraction: float = Field(default=0.2)
    seed: int = Field(default=42, description="Random seed for train/val split and fine-tuning.")
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=False)

    @classmethod
    def from_search_config(cls, values: dict[str, Any] | BaseModel | None) -> Self | None:
        if isinstance(values, BaseModel):
            return cls(**values.model_dump())
        if isinstance(values, dict):
            return cls(**values)
        if values is None:
            return None
        assert_never(values)


class HFModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = Field(
        "prajjwal1/bert-tiny", description="Name of the hugging face repository with transformer model."
    )
    batch_size: PositiveInt = Field(32, description="Batch size for model inference.")
    device: str | None = Field(None, description="Torch notation for CPU or CUDA.")
    bf16: bool = Field(False, description="Whether to use mixed precision training (not all devices support this).")
    fp16: bool = Field(False, description="Whether to use mixed precision training (not all devices support this).")
    tokenizer_config: TokenizerConfig = Field(default_factory=TokenizerConfig)
    trust_remote_code: bool = Field(False, description="Whether to trust the remote code when loading the model.")
    revision: str | None = Field(None, description="Revision from HF repo")

    @classmethod
    def from_search_config(cls, values: dict[str, Any] | str | BaseModel | None) -> Self:
        """Validate the model configuration.

        This classmethod is used to parse dictionaries that occur in search space configurations.

        Args:
            values: Model configuration values.

        Returns:
            Model configuration.
        """
        if values is None:
            return cls()
        if isinstance(values, BaseModel):
            return cls(**values.model_dump())
        if isinstance(values, str):
            return cls(model_name=values)
        return cls(**values)


def get_default_hfmodel_config() -> HFModelConfig:
    return HFModelConfig(model_name="prajjwal1/bert-tiny", revision="79779625a0a40f1eee8496e16056bc0d7766df22")


class CrossEncoderConfig(HFModelConfig):
    model_name: str = Field("cross-encoder/ms-marco-MiniLM-L6-v2", description="Name of the hugging face model.")
    train_head: bool = Field(
        False, description="Whether to train the head of the model. If False, LogReg will be trained."
    )
    tokenizer_config: TokenizerConfig = Field(
        default_factory=lambda: TokenizerConfig(max_length=512)
    )  # this is because sentence-transformers doesn't allow you to customize tokenizer settings properly


class EarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    val_fraction: float = Field(
        0.2,
        description=(
            "Fraction of train samples to allocate to dev set to monitor quality "
            "during training and perofrm early stopping if quality doesn't enhances."
        ),
    )
    patience: PositiveInt = Field(3, description="Maximum number of epoches to wait for quality to enhance.")
    threshold: FloatFromZeroToOne = Field(
        0.0,
        description="Minimum quality increment to count it as enhancement. Default: any incremeant is counted",
    )
    metric: Literal[tuple((SCORING_METRICS_MULTILABEL | SCORING_METRICS_MULTICLASS).keys())] | None = Field(  # type: ignore[valid-type]
        "scoring_f1", description="Metric to monitor."
    )

    @classmethod
    def from_search_config(cls, values: dict[str, Any] | BaseModel | None) -> Self:
        """Validate the model configuration.

        This classmethod is used to parse dictionaries that occur in search space configurations.

        Args:
            values: Model configuration values.

        Returns:
            Model configuration.
        """
        if values is None:
            return cls()
        if isinstance(values, BaseModel):
            return cls(**values.model_dump())
        if isinstance(values, dict):
            return cls(**values)
        assert_never(values)
