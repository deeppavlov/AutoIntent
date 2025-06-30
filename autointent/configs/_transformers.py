from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from typing_extensions import Self

from autointent.custom_types import FloatFromZeroToOne
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    padding: bool | Literal["longest", "max_length", "do_not_pad"] = True
    truncation: bool = True
    max_length: PositiveInt | None = Field(None, description="Maximum length of input sequences.")


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
            return values  # type: ignore[return-value]
        if isinstance(values, str):
            return cls(model_name=values)
        return cls(**values)


class TaskTypeEnum(Enum):
    """Enum for different types of prompts."""

    default = "default"
    classification = "classification"
    cluster = "cluster"
    query = "query"
    passage = "passage"
    sts = "sts"


class EmbedderConfig(HFModelConfig):
    model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Name of the hugging face model.")
    default_prompt: str | None = Field(
        None, description="Default prompt for the model. This is used when no task specific prompt is not provided."
    )
    classification_prompt: str | None = Field(None, description="Prompt for classifier.")
    cluster_prompt: str | None = Field(None, description="Prompt for clustering.")
    sts_prompt: str | None = Field(None, description="Prompt for finding most similar sentences.")
    query_prompt: str | None = Field(None, description="Prompt for query.")
    passage_prompt: str | None = Field(None, description="Prompt for passage.")
    similarity_fn_name: Literal["cosine", "dot", "euclidean", "manhattan"] = Field(
        "cosine", description="Name of the similarity function to use."
    )
    use_cache: bool = Field(True, description="Whether to use embeddings caching.")
    freeze: bool = Field(True, description="Whether to freeze the model parameters.")

    def get_prompt_config(self) -> dict[str, str] | None:
        """Get the prompt config for the given prompt type.

        Returns:
            The prompt config for the given prompt type.
        """
        prompts = {}
        if self.default_prompt:
            prompts[TaskTypeEnum.default.value] = self.default_prompt
        if self.classification_prompt:
            prompts[TaskTypeEnum.classification.value] = self.classification_prompt
        if self.cluster_prompt:
            prompts[TaskTypeEnum.cluster.value] = self.cluster_prompt
        if self.query_prompt:
            prompts[TaskTypeEnum.query.value] = self.query_prompt
        if self.passage_prompt:
            prompts[TaskTypeEnum.passage.value] = self.passage_prompt
        if self.sts_prompt:
            prompts[TaskTypeEnum.sts.value] = self.sts_prompt
        return prompts if len(prompts) > 0 else None

    def get_prompt(self, prompt_type: TaskTypeEnum | None) -> str | None:
        """Get the prompt type for the given task type.

        Args:
            prompt_type: Task type for which to get the prompt.

        Returns:
            The prompt for the given task type.
        """
        if prompt_type == TaskTypeEnum.classification and self.classification_prompt is not None:
            return self.classification_prompt
        if prompt_type == TaskTypeEnum.cluster and self.classification_prompt is not None:
            return self.cluster_prompt
        if prompt_type == TaskTypeEnum.query and self.query_prompt is not None:
            return self.query_prompt
        if prompt_type == TaskTypeEnum.passage and self.passage_prompt is not None:
            return self.passage_prompt
        if prompt_type == TaskTypeEnum.sts and self.sts_prompt is not None:
            return self.sts_prompt
        return self.default_prompt


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
            return values  # type: ignore[return-value]
        return cls(**values)
