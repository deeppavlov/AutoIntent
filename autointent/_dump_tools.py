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
from autointent.modules.abc import Module

ModuleSimpleAttributes = str | int | float | bool | list["ModuleSimpleAttributes"]

ModuleAttributes: TypeAlias = (
    ModuleSimpleAttributes
    | npt.NDArray[Any]
    | Embedder
    | VectorIndex
    | BaseEstimator
    | SentenceTransformer
    | CrossEncoder
)


class Dumper:
    simple_attrs = "simple_attrs.json"
    arrays = "arrays.npz"
    embedders = "embedders"
    indexes = "vector_indexes"
    estimators = "estimators"
    sentence_transformers = "sentence_transformers"
    cross_encoders = "cross_encoders"

    @staticmethod
    def dump(module: Module, path: Path) -> None:
        """Dump modules attributes to filestystem."""
        attrs: dict[str, ModuleAttributes] = vars(module)
        simple_attrs = {}
        arrays: dict[str, npt.NDArray[Any]] = {}

        for key, val in attrs.items():
            if isinstance(val, ModuleSimpleAttributes):
                simple_attrs[key] = val
            elif isinstance(val, np.ndarray):
                arrays[key] = val
            elif isinstance(val, Embedder):
                val.dump(path / Dumper.embedders / key)
            elif isinstance(val, VectorIndex):
                val.dump(path / Dumper.indexes / key)
            elif isinstance(val, BaseEstimator):
                joblib.dump(val, path / Dumper.estimators / key)
            elif isinstance(val, SentenceTransformer):
                val.save(str(path / Dumper.sentence_transformers / key))
            elif isinstance(val, CrossEncoder):
                val.save(str(path / Dumper.cross_encoders / key))
            else:
                msg = f"Attribute {key} of type {type(val)} cannot be dumped to file system."
                raise TypeError(msg)

        with (path / Dumper.simple_attrs).open("w") as file:
            json.dump(simple_attrs, file, ensure_ascii=False, indent=4)

        np.savez(path / Dumper.arrays, allow_pickle=False, **arrays)

    @staticmethod
    def load(module: Module, path: Path) -> None:
        """Load attributes from file system."""
        for child in path.iterdir():
            if child.name == Dumper.simple_attrs:
                with child.open() as file:
                    simple_attrs = json.load(file)
            elif child.name == Dumper.arrays:
                arrays = np.load(child)
            elif child.name == Dumper.embedders:
                # TODO propagate custom loading params (such as device, batch size etc) to this line
                embedders = {embedder_dump.name: Embedder.load(embedder_dump) for embedder_dump in child.iterdir()}
            elif child.name == Dumper.indexes:
                indexes = {index_dump.name: VectorIndex.load(index_dump) for index_dump in child.iterdir()}
            elif child.name == Dumper.estimators:
                estimators = {estimator_dump.name: joblib.load(estimator_dump) for estimator_dump in child.iterdir()}
            elif child.name == Dumper.sentence_transformers:
                sentence_transformers = {
                    transformer_dump.name: SentenceTransformer(transformer_dump) for transformer_dump in child.iterdir()
                }
            elif child.name == Dumper.cross_encoders:
                cross_encoders = {
                    cross_encoder_dump.name: CrossEncoder(cross_encoder_dump) for cross_encoder_dump in child.iterdir()
                }
            else:
                msg = f"Found unexpected child {child}"
                raise ValueError(msg)
        module.__dict__.update(
            simple_attrs | arrays | embedders | indexes | estimators | sentence_transformers | cross_encoders
        )
