import logging
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import numpy as np
import numpy.typing as npt

from autointent.configs import CrossEncoderConfig, EmbedderConfig
from autointent.context.optimization_info import Artifact
from autointent.schemas import TagsList

from .base import BaseObjectDumper, ModuleAttributes, ModuleSimpleAttributes
from .generator_dumper import GeneratorDumper
from .unit_dumpers import (
    ArraysDumper,
    CatBoostDumper,
    EmbedderDumper,
    EstimatorDumper,
    HFModelDumper,
    HFTokenizerDumper,
    PeftModelDumper,
    PydanticModelDumper,
    RankerDumper,
    SimpleAttributesDumper,
    TagsListDumper,
    TorchModelDumper,
    VectorIndexDumper,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class Dumper:
    # List of all available dumper classes
    _DUMPER_CLASSES: ClassVar[list[type[BaseObjectDumper[Any]]]] = [
        TagsListDumper,
        SimpleAttributesDumper,
        ArraysDumper,
        EmbedderDumper,
        VectorIndexDumper,
        EstimatorDumper,
        RankerDumper,
        PydanticModelDumper,
        PeftModelDumper,
        HFModelDumper,
        HFTokenizerDumper,
        TorchModelDumper,
        CatBoostDumper,
        GeneratorDumper,
    ]

    @staticmethod
    def _get_dumper_for_object(obj: Any) -> type[BaseObjectDumper[Any]] | None:  # noqa: ANN401
        """Get the appropriate dumper class for an object."""
        for dumper_class in Dumper._DUMPER_CLASSES:
            if dumper_class.check_isinstance(obj):
                return dumper_class
        return None

    @staticmethod
    def _dump_single_object(key: str, obj: Any, path: Path, exists_ok: bool, raise_errors: bool) -> None:  # noqa: ANN401
        """Dump a single object using the appropriate dumper."""
        dumper_class = Dumper._get_dumper_for_object(obj)
        if dumper_class is None:
            msg = f"Attribute {key} of type {type(obj)} cannot be dumped to file system."
            logger.error(msg)
            if raise_errors:
                raise TypeError(msg)
            return

        try:
            dumper_path = path / dumper_class.dir_or_file_name / key
            dumper_class.dump(obj, dumper_path, exists_ok)
        except Exception as e:
            msg = f"Error dumping {key} of type {type(obj)}: {e}"
            logger.exception(msg)
            if raise_errors:
                raise

    @staticmethod
    def dump(
        obj: Any,  # noqa: ANN401
        path: Path,
        exists_ok: bool = False,
        exclude: list[type[Any]] | None = None,
        raise_errors: bool = False,
    ) -> None:
        """Dump modules attributes to filestystem.

        Args:
            obj: Object to dump
            path: Path to dump to
            exists_ok: If True, do not raise an error if the directory already exists
            exclude: List of types to exclude from dumping
            raise_errors: whether to raise dumping errors or just log
        """
        attrs: dict[str, ModuleAttributes] = vars(obj)
        simple_attrs: dict[str, ModuleSimpleAttributes] = {}
        arrays: dict[str, npt.NDArray[Any]] = {}

        for key, val in attrs.items():
            if isinstance(val, Artifact) or (exclude and isinstance(val, tuple(exclude))):
                continue

            # Handle simple attributes and arrays separately
            if isinstance(val, ModuleSimpleAttributes) and not isinstance(val, TagsList):
                simple_attrs[key] = val
            elif isinstance(val, np.ndarray):
                arrays[key] = val
            else:
                # Use the appropriate dumper for complex objects
                Dumper._dump_single_object(key, val, path, exists_ok, raise_errors)

        # Dump simple attributes and arrays
        if simple_attrs:
            SimpleAttributesDumper.dump(simple_attrs, path / SimpleAttributesDumper.dir_or_file_name, exists_ok)
        if arrays:
            ArraysDumper.dump(arrays, path / ArraysDumper.dir_or_file_name, exists_ok)

    @staticmethod
    def _load_objects_from_directory(
        path: Path,
        dumper_class: type[BaseObjectDumper[T]],
        raise_errors: bool,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, T]:
        """Load all objects from a directory using the specified dumper."""
        res: dict[str, T] = {}
        if not path.exists():
            return res

        for item_path in path.iterdir():
            try:
                res[item_path.name] = dumper_class.load(item_path, **kwargs)
            except Exception as e:  # noqa: PERF203
                msg = f"Error loading {item_path.name} using {dumper_class.__name__}: {e}"
                logger.exception(msg)
                if raise_errors:
                    raise

        return res

    @staticmethod
    def _load_object_from_file(
        path: Path,
        dumper_class: type[BaseObjectDumper[dict[str, Any]]],
        raise_errors: bool,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """Load objects from a single file using the specified dumper."""
        res = {}
        if path.exists():
            try:
                res = dumper_class.load(path, **kwargs)
            except Exception as e:
                msg = f"Error loading from {path} using {dumper_class.__name__}: {e}"
                logger.exception(msg)
                if raise_errors:
                    raise
        return res

    @staticmethod
    def load(
        obj: Any,  # noqa: ANN401
        path: Path,
        embedder_config: EmbedderConfig | None = None,
        cross_encoder_config: CrossEncoderConfig | None = None,
        raise_errors: bool = False,
    ) -> None:
        """Load attributes from file system."""
        loaded_attrs: dict[str, Any] = {}

        for dumper_class in Dumper._DUMPER_CLASSES:
            if dumper_class in [SimpleAttributesDumper, ArraysDumper]:
                dumper_func = Dumper._load_object_from_file
            else:
                dumper_func = Dumper._load_objects_from_directory

            dir_path = path / dumper_class.dir_or_file_name

            try:
                objects = dumper_func(
                    dir_path,
                    dumper_class,
                    raise_errors,
                    cross_encoder_config=cross_encoder_config,
                    embedder_config=embedder_config,
                )
                loaded_attrs.update(objects)
            except Exception as e:
                msg = f"Error loading objects from {dir_path} using {dumper_class.__name__}: {e}"
                logger.exception(msg)
                if raise_errors:
                    raise

        obj.__dict__.update(loaded_attrs)
