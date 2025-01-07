import json
from pathlib import Path
from typing import Any, TypeAlias

import joblib
import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.base import BaseEstimator

from autointent import Embedder
from autointent.context.vector_index_client import VectorIndex

ModuleSimpleAttributes = str | int | float | bool | list[Any]

ModuleAttributes: TypeAlias = (
    ModuleSimpleAttributes
    | npt.NDArray[Any]
    | Embedder
    | VectorIndex
    | BaseEstimator
    | SentenceTransformer
    | CrossEncoder
)


def dump_attrs(attrs: dict[str, ModuleAttributes], path: Path) -> None:
    """Dump modules attributes to filestystem."""
    simple_attrs = {}
    arrays: dict[str, npt.NDArray[Any]] = {}

    simple_attrs_path = path / "simple_attrs.json"
    arrays_path = path / "arrays"
    embedders_path = path / "embedders"
    indexes_path = path / "vector_indexes"
    estimators_path = path / "estimators"
    sentence_transformers_path = path / "sentence_transformers"
    cross_encoders_path = path / "cross_encoders"

    for key, val in attrs.items():
        if isinstance(val, ModuleSimpleAttributes):
            simple_attrs[key] = val
        elif isinstance(val, np.ndarray):
            arrays[key] = val
        elif isinstance(val, Embedder):
            val.dump(embedders_path / key)
        elif isinstance(val, VectorIndex):
            val.dump(indexes_path / key)
        elif isinstance(val, BaseEstimator):
            joblib.dump(val, estimators_path / key)
        elif isinstance(val, SentenceTransformer):
            val.save(str(sentence_transformers_path / key))
        elif isinstance(val, CrossEncoder):
            val.save(str(cross_encoders_path / key))
        else:
            msg = f"Attribute {key} of type {type(val)} cannot be dumped to file system."
            raise TypeError(msg)

    with simple_attrs_path.open("w") as file:
        json.dump(simple_attrs, file, ensure_ascii=False, indent=4)

    np.savez(arrays_path, allow_pickle=False, **arrays)
