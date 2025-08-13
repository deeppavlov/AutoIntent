from typing import Any, TypedDict

from pydantic import BaseModel


class VectorIndexConfig(BaseModel):
    vector_size: int


class FaissConfig(VectorIndexConfig): ...


class OpenSearchHost(TypedDict):
    host: str
    port: int


class OpenSearchConfig(VectorIndexConfig):
    hosts: list[OpenSearchHost]
    kwargs: dict[str, Any]  # TODO define set of options
