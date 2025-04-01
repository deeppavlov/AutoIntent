import inspect
import json
import logging
import types
from pathlib import Path
from types import UnionType
from typing import Any, TypeAlias, Union, get_args, get_origin

import joblib
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from autointent import Embedder, Ranker, VectorIndex
from autointent.configs import CrossEncoderConfig, EmbedderConfig
from autointent.schemas import TagsList

ModuleSimpleAttributes = None | str | int | float | bool | list  # type: ignore[type-arg]

ModuleAttributes: TypeAlias = (
    ModuleSimpleAttributes
    | TagsList
    | npt.NDArray[Any]
    | Embedder
    | VectorIndex
    | BaseEstimator
    | Ranker
    | BaseModel
    | PreTrainedModel
    | PreTrainedTokenizer
    | PreTrainedTokenizerFast
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
    hf_models = "hf_models"
    hf_tokenizers = "hf_tokenizers"

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
            path / Dumper.hf_models,
            path / Dumper.hf_tokenizers,
        ]
        for subdir in subdirectories:
            subdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def dump(obj: Any, path: Path) -> None:  # noqa: ANN401, C901, PLR0912
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
            if isinstance(val, PreTrainedModel):
                try:
                    model_path = path / Dumper.hf_models / key
                    val.save_pretrained(model_path)
                except Exception:
                    logger.exception("Error dumping Hugging Face model %s", key)
            elif isinstance(val, PreTrainedTokenizer | PreTrainedTokenizerFast):
                try:
                    tokenizer_path = path / Dumper.hf_tokenizers / key
                    val.save_pretrained(tokenizer_path)
                except Exception:
                    logger.exception("Error dumping Hugging Face tokenizer %s", key)
            elif isinstance(val, BaseModel):
                try:
                    pydantic_path = path / Dumper.pydantic_models / f"{key}.json"
                    with pydantic_path.open("w", encoding="utf-8") as file:
                        json.dump(val.model_dump(), file, ensure_ascii=False, indent=4)
                except Exception:
                    logger.exception("Error dumping pydantic model %s", key)
            elif isinstance(val, TagsList):
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
                try:
                    joblib.dump(val, path / Dumper.estimators / f"{key}.joblib")
                except Exception:
                    logger.exception("Error dumping BaseEstimator %s", key)
            elif isinstance(val, Ranker):
                val.save(str(path / Dumper.cross_encoders / key))
            elif not isinstance(val, type | types.ModuleType | types.FunctionType | types.MethodType):
                logger.warning("Attribute '%s' of type %s cannot be dumped and will be skipped.", key, type(val))

        with (path / Dumper.simple_attrs).open("w", encoding="utf-8") as file:
            json.dump(simple_attrs, file, ensure_ascii=False, indent=4)

        if arrays:
            try:
                np.savez(path / Dumper.arrays, allow_pickle=False, **arrays)
            except Exception:
                logger.exception("Error saving numpy arrays to %s", path / Dumper.arrays)

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
        hf_models: dict[str, Any] = {}
        hf_tokenizers: dict[str, Any] = {}

        for child in path.iterdir():
            if child.is_file():
                if child.name == Dumper.simple_attrs:
                    try:
                        with child.open(encoding="utf-8") as file:
                            simple_attrs = json.load(file)
                    except Exception:
                        logger.exception("Error loading simple attributes from %s", child)
                elif child.name == Dumper.arrays:
                    try:
                        arrays = dict(np.load(child, allow_pickle=False))
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Could not load numpy arrays from %s: %s", child, e)

            elif child.is_dir():
                if child.name == Dumper.hf_models:
                    for model_dir in child.iterdir():
                        if model_dir.is_dir():
                            attr_name = model_dir.name
                            try:
                                hf_models[attr_name] = AutoModelForSequenceClassification.from_pretrained(model_dir)
                            except Exception:
                                logger.exception("Error loading Hugging Face model '%s' from %s", attr_name, model_dir)
                elif child.name == Dumper.hf_tokenizers:
                    for tokenizer_dir in child.iterdir():
                        if tokenizer_dir.is_dir():
                            attr_name = tokenizer_dir.name
                            try:
                                hf_tokenizers[attr_name] = AutoTokenizer.from_pretrained(tokenizer_dir)
                            except Exception:
                                logger.exception(
                                    "Error loading Hugging Face tokenizer '%s' from %s", attr_name, tokenizer_dir
                                )
                elif child.name == Dumper.pydantic_models:
                    for model_file in child.iterdir():
                        if model_file.is_file() and model_file.suffix == ".json":
                            variable_name = model_file.stem
                            try:
                                with model_file.open("r", encoding="utf-8") as file:
                                    content = json.load(file)

                                model_type = obj.__class__.__annotations__.get(variable_name)

                                if model_type is None:
                                    sig = inspect.signature(obj.__init__)
                                    if variable_name in sig.parameters:
                                        model_type = sig.parameters[variable_name].annotation

                                if model_type is None:
                                    logger.error("No type annotation found for pydantic model %s", variable_name)
                                    continue

                                potential_types: list[Any] = []  # Added type annotation
                                if get_origin(model_type) in (UnionType, Union):
                                    potential_types.extend(get_args(model_type))
                                else:
                                    potential_types.append(model_type)

                                pydantic_type = None
                                for p_type in potential_types:
                                    if inspect.isclass(p_type) and issubclass(p_type, BaseModel):
                                        pydantic_type = p_type
                                        break

                                if pydantic_type is None:
                                    logger.error("No pydantic type found in annotation for %s", variable_name)
                                    continue

                                pydantic_models[variable_name] = pydantic_type(**content)
                            except Exception:
                                logger.exception("Error loading pydantic model %s from %s", variable_name, model_file)

                elif child.name == Dumper.tags:
                    tags = {tags_dump.name: TagsList.load(tags_dump) for tags_dump in child.iterdir()}
                elif child.name == Dumper.embedders:
                    embedders = {
                        embedder_dump.name: Embedder.load(embedder_dump, override_config=embedder_config)
                        for embedder_dump in child.iterdir()
                    }
                elif child.name == Dumper.indexes:
                    indexes = {index_dump.name: VectorIndex.load(index_dump) for index_dump in child.iterdir()}
                elif child.name == Dumper.estimators:
                    estimators = {}
                    for estimator_dump in child.iterdir():
                        if estimator_dump.is_file() and estimator_dump.suffix == ".joblib":
                            try:
                                estimators[estimator_dump.stem] = joblib.load(estimator_dump)
                            except Exception:
                                logger.exception(
                                    "Error loading estimator %s from %s", estimator_dump.stem, estimator_dump
                                )
                elif child.name == Dumper.cross_encoders:
                    cross_encoders = {
                        cross_encoder_dump.name: Ranker.load(cross_encoder_dump, override_config=cross_encoder_config)
                        for cross_encoder_dump in child.iterdir()
                    }

        obj.__dict__.update(
            tags
            | simple_attrs
            | arrays
            | embedders
            | indexes
            | estimators
            | cross_encoders
            | pydantic_models
            | hf_models
            | hf_tokenizers
        )
