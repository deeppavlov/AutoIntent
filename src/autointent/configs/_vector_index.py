from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class VectorIndexConfig(BaseModel): ...


class FaissConfig(VectorIndexConfig): ...


class OpenSearchHost(TypedDict):
    host: str
    port: int


class OpenSearchConfig(VectorIndexConfig):
    hosts: list[OpenSearchHost]
    index_name: str | None = None
    init_kwargs: dict[str, Any] = Field(default_factory=dict)  # TODO define set of options


def get_default_vector_index_config() -> VectorIndexConfig:
    return FaissConfig()
