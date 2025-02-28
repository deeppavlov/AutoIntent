import inspect
import json
import logging
from pathlib import Path
from types import UnionType
from typing import Any, TypeAlias, Union, get_args, get_origin

import joblib
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from sklearn.base import BaseEstimator

from autointent import Embedder, Ranker, VectorIndex
from autointent.configs import CrossEncoderConfig, EmbedderConfig
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
    pydantic_models: str = "pydantic"

    @staticmethod
    def make_subdirectories(path: Path) -> None:
        """Make subdirectories for dumping.

        Args:
            path: Path to make subdirectories in
        """
        subdirectories = [
            path / Dumper.tags,
            path / Dumper.embedders,
            path / Dumper.indexes,
            path / Dumper.estimators,
            path / Dumper.cross_encoders,
            path / Dumper.pydantic_models,
        ]
        for subdir in subdirectories:
            subdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def dump(obj: Any, path: Path) -> None:  # noqa: ANN401, C901
        """Dump modules attributes to filestystem.

        Args:
            obj: Object to dump
            path: Path to dump to
        """
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
            elif isinstance(val, CrossEncoderConfig | EmbedderConfig):
                try:
                    pydantic_path = path / Dumper.pydantic_models / f"{key}.json"
                    with pydantic_path.open("w", encoding="utf-8") as file:
                        json.dump(val.model_dump(), file, ensure_ascii=False, indent=4)
                except Exception as e:
                    msg = f"Error dumping pydantic model {key}: {e}"
                    logging.exception(msg)
            else:
                msg = f"Attribute {key} of type {type(val)} cannot be dumped to file system."
                logger.error(msg)

        with (path / Dumper.simple_attrs).open("w") as file:
            json.dump(simple_attrs, file, ensure_ascii=False, indent=4)

        np.savez(path / Dumper.arrays, allow_pickle=False, **arrays)

    @staticmethod
    def load(  # noqa: PLR0912, C901, PLR0915
        obj: Any,  # noqa: ANN401
        path: Path,
        embedder_config: EmbedderConfig | None = None,
        cross_encoder_config: CrossEncoderConfig | None = None,
    ) -> None:
        """Load attributes from file system."""
        tags: dict[str, Any] = {}
        simple_attrs: dict[str, Any] = {}
        arrays: dict[str, Any] = {}
        embedders: dict[str, Any] = {}
        indexes: dict[str, Any] = {}
        estimators: dict[str, Any] = {}
        cross_encoders: dict[str, Any] = {}
        pydantic_models: dict[str, Any] = {}

        for child in path.iterdir():
            if child.name == Dumper.tags:
                tags = {tags_dump.name: TagsList.load(tags_dump) for tags_dump in child.iterdir()}
            elif child.name == Dumper.simple_attrs:
                with child.open() as file:
                    simple_attrs = json.load(file)
            elif child.name == Dumper.arrays:
                arrays = dict(np.load(child))
            elif child.name == Dumper.embedders:
                embedders = {
                    embedder_dump.name: Embedder.load(embedder_dump, override_config=embedder_config)
                    for embedder_dump in child.iterdir()
                }
            elif child.name == Dumper.indexes:
                indexes = {index_dump.name: VectorIndex.load(index_dump) for index_dump in child.iterdir()}
            elif child.name == Dumper.estimators:
                estimators = {estimator_dump.name: joblib.load(estimator_dump) for estimator_dump in child.iterdir()}
            elif child.name == Dumper.cross_encoders:
                cross_encoders = {
                    cross_encoder_dump.name: Ranker.load(cross_encoder_dump, override_config=cross_encoder_config)
                    for cross_encoder_dump in child.iterdir()
                }
            elif child.name == Dumper.pydantic_models:
                for model_file in child.iterdir():
                    with model_file.open("r", encoding="utf-8") as file:
                        content = json.load(file)
                    variable_name = model_file.stem

                    # First try to get the type annotation from the class annotations.
                    model_type = obj.__class__.__annotations__.get(variable_name)

                    # Fallback: inspect __init__ signature if not found in class-level annotations.
                    if model_type is None:
                        sig = inspect.signature(obj.__init__)
                        if variable_name in sig.parameters:
                            model_type = sig.parameters[variable_name].annotation

                    if model_type is None:
                        msg = f"No type annotation found for {variable_name}"
                        logger.error(msg)
                        continue

                    # If the annotation is a Union, extract the pydantic model type.
                    if get_origin(model_type) in (UnionType, Union):
                        for arg in get_args(model_type):
                            if isinstance(arg, type) and issubclass(arg, BaseModel):
                                model_type = arg
                                break
                        else:
                            msg = f"No pydantic type found in Union for {variable_name}"
                            logger.error(msg)
                            continue

                    if not (isinstance(model_type, type) and issubclass(model_type, BaseModel)):
                        msg = f"Type for {variable_name} is not a pydantic model: {model_type}"
                        logger.error(msg)
                        continue

                    pydantic_models[variable_name] = model_type(**content)
            else:
                msg = f"Found unexpected child {child}"
                logger.error(msg)
        obj.__dict__.update(
            tags | simple_attrs | arrays | embedders | indexes | estimators | cross_encoders | pydantic_models
        )
