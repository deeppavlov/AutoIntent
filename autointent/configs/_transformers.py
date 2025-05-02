from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from typing_extensions import Self, assert_never


class TokenizerConfig(BaseModel):
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
    tokenizer_config: TokenizerConfig = Field(default_factory=TokenizerConfig)
    trust_remote_code: bool = Field(False, description="Whether to trust the remote code when loading the model.")

    @classmethod
    def from_search_config(cls, values: dict[str, Any] | str | BaseModel | None) -> Self:
        """Validate the model configuration.

        Args:
            values: Model configuration values. If a string is provided, it is converted to a dictionary.

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
    classifier_prompt: str | None = Field(None, description="Prompt for classifier.")
    cluster_prompt: str | None = Field(None, description="Prompt for clustering.")
    sts_prompt: str | None = Field(None, description="Prompt for finding most similar sentences.")
    query_prompt: str | None = Field(None, description="Prompt for query.")
    passage_prompt: str | None = Field(None, description="Prompt for passage.")
    similarity_fn_name: str | None = Field(
        "cosine", description="Name of the similarity function to use (cosine, dot, euclidean, manhattan)."
    )

    def get_prompt_config(self) -> dict[str, str] | None:
        """Get the prompt config for the given prompt type.

        Returns:
            The prompt config for the given prompt type.
        """
        prompts = {}
        if self.default_prompt:
            prompts[TaskTypeEnum.default.value] = self.default_prompt
        if self.classifier_prompt:
            prompts[TaskTypeEnum.classification.value] = self.classifier_prompt
        if self.cluster_prompt:
            prompts[TaskTypeEnum.cluster.value] = self.cluster_prompt
        if self.query_prompt:
            prompts[TaskTypeEnum.query.value] = self.query_prompt
        if self.passage_prompt:
            prompts[TaskTypeEnum.passage.value] = self.passage_prompt
        if self.sts_prompt:
            prompts[TaskTypeEnum.sts.value] = self.sts_prompt
        return prompts if len(prompts) > 0 else None

    def get_prompt_type(self, prompt_type: TaskTypeEnum | None) -> str | None:  # noqa: PLR0911
        """Get the prompt type for the given task type.

        Args:
            prompt_type: Task type for which to get the prompt.

        Returns:
            The prompt for the given task type.
        """
        if prompt_type is None:
            return self.default_prompt
        if prompt_type == TaskTypeEnum.classification:
            return self.classifier_prompt
        if prompt_type == TaskTypeEnum.cluster:
            return self.cluster_prompt
        if prompt_type == TaskTypeEnum.query:
            return self.query_prompt
        if prompt_type == TaskTypeEnum.passage:
            return self.passage_prompt
        if prompt_type == TaskTypeEnum.sts:
            return self.sts_prompt
        if prompt_type == TaskTypeEnum.default:
            return self.default_prompt
        assert_never(prompt_type)

    use_cache: bool = Field(False, description="Whether to use embeddings caching.")


class CrossEncoderConfig(HFModelConfig):
    model_name: str = Field("cross-encoder/ms-marco-MiniLM-L6-v2", description="Name of the hugging face model.")
    train_head: bool = Field(
        False, description="Whether to train the head of the model. If False, LogReg will be trained."
    )
    tokenizer_config: TokenizerConfig = Field(
        default_factory=lambda: TokenizerConfig(max_length=512)
    )  # this is because sentence-transformers doesn't allow you to customize tokenizer settings properly
