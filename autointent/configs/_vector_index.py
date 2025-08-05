from typing import Any, TypedDict

from pydantic import BaseModel


class FaissConfig(BaseModel): ...


class OpenSearchHost(TypedDict):
    host: str
    port: int


class OpenSearchConfig(BaseModel):
    hosts: list[OpenSearchHost]
    kwargs: dict[str, Any]  # TODO define set of options


class VectorIndexConfig(BaseModel):
    vector_size: int
    faiss: FaissConfig
    opensearch: OpenSearchConfig
