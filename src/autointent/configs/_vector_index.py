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
    index_name: str | None = Field(
        None,
        description=(
            "Name of the OpenSearch index. AutoIntent takes ownership of this index during fit(): "
            "the first write of each fit clears its contents (fit-replaces semantics), so do not "
            "point it at a collection you want to keep. To query an existing collection without "
            "modifying it, use a loaded pipeline (load() + predict only). If None, a name is "
            "derived from the first document added."
        ),
    )
    init_kwargs: dict[str, Any] = Field(default_factory=dict)  # TODO define set of options


def get_default_vector_index_config() -> VectorIndexConfig:
    return FaissConfig()
