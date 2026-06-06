from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from typing_extensions import assert_never

from autointent.custom_types import FloatFromZeroToOne
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL

if TYPE_CHECKING:
    from typing_extensions import Self


# Pinned commit SHAs for the Hugging Face models that ship as defaults in
# autointent. When a config is constructed with one of these model_name values
# and no explicit ``revision``, the SHA below is filled in automatically so the
# library never has to call the HF API just to resolve ``main`` to a hash for
# cache keying. Update an entry here when you intentionally want to move a
# default to a newer revision.
DEFAULT_REVISIONS: dict[str, str] = {
    "prajjwal1/bert-tiny": "79779625a0a40f1eee8496e16056bc0d7766df22",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "intfloat/multilingual-e5-large-instruct": "274baa43b0e13e37fafa6428dbc7938e62e5c439",
    "intfloat/multilingual-e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
    "avsolatorio/GIST-small-Embedding-v0": "75e62fd210b9fde790430e0b2f040b0b00a021b1",
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    # Used heavily in the embedder test suite (tests/embedder/conftest.py).
    # Pinning the SHA here lets HFModelConfig auto-fill it via the validator
    # so sentence-transformers never asks the Hub for "main" — which 429s
    # under parallel matrix load even when the model files are cached.
    "sergeyzh/rubert-tiny-turbo": "93769a3baad2b037e5c2e4312fccf6bcfe082bf1",
}


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

    @model_validator(mode="after")
    def _apply_default_revision(self) -> Self:
        if self.revision is None and self.model_name in DEFAULT_REVISIONS:
            self.revision = DEFAULT_REVISIONS[self.model_name]
        return self

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
    return HFModelConfig()


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
