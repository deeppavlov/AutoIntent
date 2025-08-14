from typing import Any, TypedDict

from pydantic import BaseModel, Field


class VectorIndexConfig(BaseModel): ...


class FaissConfig(VectorIndexConfig): ...


class OpenSearchHost(TypedDict):
    host: str
    port: int


class OpenSearchConfig(VectorIndexConfig):
    hosts: list[OpenSearchHost]
    index_name: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)  # TODO define set of options
