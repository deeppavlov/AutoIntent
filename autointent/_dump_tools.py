import json
import logging
from pathlib import Path
from typing import Any, TypeAlias

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from autointent import Embedder, Ranker, VectorIndex
from autointent.schemas import TagsList

ModuleSimpleAttributes = None | str | int | float | bool | list  # type: ignore[type-arg]

ModuleAttributes: TypeAlias = (
    ModuleSimpleAttributes | TagsList | np.ndarray | Embedder | VectorIndex | BaseEstimator | Ranker  # type: ignore[type-arg]
)

logger = logging.getLogger(__name__)


class Dumper:
    tags = "tags"
    simple_attrs = "simple_attrs.json"
    arrays = "arrays.npz"
    embedders = "embedders"
    indexes = "vector_indexes"
    estimators = "estimators"
    cross_encoders = "cross_encoders"

    @staticmethod
    def make_subdirectories(path: Path) -> None:
        subdirectories = [
            path / Dumper.tags,
            path / Dumper.embedders,
            path / Dumper.indexes,
            path / Dumper.estimators,
            path / Dumper.cross_encoders,
        ]
        for subdir in subdirectories:
            subdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def dump(obj: Any, path: Path) -> None:  # noqa: ANN401
        """Dump modules attributes to filestystem."""
        attrs: dict[str, ModuleAttributes] = vars(obj)
        simple_attrs = {}
        arrays: dict[str, npt.NDArray[Any]] = {}

        Dumper.make_subdirectories(path)

        for key, val in attrs.items():
            if isinstance(val, TagsList):
                val.dump(path / Dumper.tags / key)
            elif isinstance(val, ModuleSimpleAttributes):
                simple_attrs[key] = val
            elif isinstance(val, np.ndarray):
                arrays[key] = val
            elif isinstance(val, Embedder):
                val.dump(path / Dumper.embedders / key)
            elif isinstance(val, VectorIndex):
                val.dump(path / Dumper.indexes / key)
            elif isinstance(val, BaseEstimator):
                joblib.dump(val, path / Dumper.estimators / key)
            elif isinstance(val, Ranker):
                val.save(str(path / Dumper.cross_encoders / key))
            else:
                msg = f"Attribute {key} of type {type(val)} cannot be dumped to file system."
                logger.error(msg)

        with (path / Dumper.simple_attrs).open("w") as file:
            json.dump(simple_attrs, file, ensure_ascii=False, indent=4)

        np.savez(path / Dumper.arrays, allow_pickle=False, **arrays)

    @staticmethod
    def load(obj: Any, path: Path) -> None:  # noqa: ANN401
        """Load attributes from file system."""
        for child in path.iterdir():
            if child.name == Dumper.tags:
                tags = {tags_dump.name: TagsList.load(tags_dump) for tags_dump in child.iterdir()}
            elif child.name == Dumper.simple_attrs:
                with child.open() as file:
                    simple_attrs = json.load(file)
            elif child.name == Dumper.arrays:
                arrays = dict(np.load(child))
            elif child.name == Dumper.embedders:
                # TODO propagate custom loading params (such as device, batch size etc) to this line
                embedders = {embedder_dump.name: Embedder.load(embedder_dump) for embedder_dump in child.iterdir()}
            elif child.name == Dumper.indexes:
                indexes = {index_dump.name: VectorIndex.load(index_dump) for index_dump in child.iterdir()}
            elif child.name == Dumper.estimators:
                estimators = {estimator_dump.name: joblib.load(estimator_dump) for estimator_dump in child.iterdir()}
            elif child.name == Dumper.cross_encoders:
                cross_encoders = {
                    cross_encoder_dump.name: Ranker.load(cross_encoder_dump) for cross_encoder_dump in child.iterdir()
                }
            else:
                msg = f"Found unexpected child {child}"
                logger.error(msg)
        obj.__dict__.update(tags | simple_attrs | arrays | embedders | indexes | estimators | cross_encoders)
