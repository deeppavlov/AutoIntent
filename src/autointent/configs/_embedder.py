from abc import ABC
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ._transformers import HFModelConfig


class TaskTypeEnum(Enum):
    """Enum for different types of prompts."""

    default = "default"
    classification = "classification"
    cluster = "cluster"
    query = "query"
    passage = "passage"
    sts = "sts"


class EmbedderConfig(ABC, BaseModel, extra="forbid"):
    """Base class for embedder configurations."""

    default_prompt: str | None = Field(
        None, description="Default prompt for the model. This is used when no task specific prompt is not provided."
    )
    classification_prompt: str | None = Field(None, description="Prompt for classifier.")
    cluster_prompt: str | None = Field(None, description="Prompt for clustering.")
    sts_prompt: str | None = Field(None, description="Prompt for finding most similar sentences.")
    query_prompt: str | None = Field(None, description="Prompt for query.")
    passage_prompt: str | None = Field(None, description="Prompt for passage.")
    use_cache: bool = Field(True, description="Whether to use embeddings caching.")

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

    def get_prompt(self, prompt_type: "TaskTypeEnum | None") -> str | None:
        """Get the prompt type for the given task type.

        Args:
            prompt_type: Task type for which to get the prompt.

        Returns:
            The prompt for the given task type.
        """
        if prompt_type == TaskTypeEnum.classification and self.classification_prompt is not None:
            return self.classification_prompt
        if prompt_type == TaskTypeEnum.cluster and self.cluster_prompt is not None:
            return self.cluster_prompt
        if prompt_type == TaskTypeEnum.query and self.query_prompt is not None:
            return self.query_prompt
        if prompt_type == TaskTypeEnum.passage and self.passage_prompt is not None:
            return self.passage_prompt
        if prompt_type == TaskTypeEnum.sts and self.sts_prompt is not None:
            return self.sts_prompt
        return self.default_prompt


class SentenceTransformerEmbeddingConfig(EmbedderConfig, HFModelConfig):
    """Configuration for Sentence Transformer based embeddings."""

    model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Name of the hugging face model.")
    similarity_fn_name: str | None = Field(
        None, description="Name of the similarity function to use. Set to `None` to use model-native."
    )


class OpenaiEmbeddingConfig(EmbedderConfig):
    """Configuration for OpenAI based embeddings."""

    model_name: str = Field("text-embedding-3-small", description="Name of the OpenAI embedding model.")
    api_key: str = Field(description="OpenAI API key. If None, will look for OPENAI_API_KEY environment variable.")
    base_url: str | None = Field(default=None, description="Base URL for OpenAI API calls")
    batch_size: int = Field(100, description="Batch size for API requests.")
    max_retries: int = Field(3, description="Maximum number of retries for failed API requests.")
    timeout: float = Field(30.0, description="Timeout for API requests in seconds.")
    dimensions: int | None = Field(
        None, description="Number of dimensions for the embedding. Only supported for certain models."
    )
    max_concurrent: int | None = Field(
        None, description="Maximum number of concurrent API requests. If None, uses synchronous processing."
    )
    max_per_second: float | None = Field(
        None, description="Maximum number of API requests per second. Only used with async processing."
    )


def get_default_embedder_config(**kwargs: Any) -> EmbedderConfig:  # noqa: ANN401
    return SentenceTransformerEmbeddingConfig.model_validate(kwargs)


def initialize_embedder_config(values: dict[str, Any] | str | EmbedderConfig | None) -> EmbedderConfig:
    if values is None:
        return get_default_embedder_config()
    if isinstance(values, EmbedderConfig):
        return values.model_copy(deep=True)
    if isinstance(values, str):
        return get_default_embedder_config(model_name=values)
    return get_default_embedder_config(**values)
