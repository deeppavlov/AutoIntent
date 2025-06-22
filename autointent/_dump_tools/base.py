import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypeVar

import numpy as np
from sklearn.base import BaseEstimator

from autointent import Embedder, Ranker, VectorIndex
from autointent._wrappers import BaseTorchModuleWithVocab
from autointent.schemas import TagsList

ModuleSimpleAttributes = None | str | int | float | bool | list  # type: ignore[type-arg]

ModuleAttributes: TypeAlias = (
    ModuleSimpleAttributes
    | TagsList
    | np.ndarray  # type: ignore[type-arg]
    | Embedder
    | VectorIndex
    | BaseEstimator
    | Ranker
    | BaseTorchModuleWithVocab
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseObjectDumper(ABC, Generic[T]):
    dir_or_file_name: str

    @staticmethod
    @abstractmethod
    def dump(obj: T, path: Path, exists_ok: bool) -> None: ...

    @staticmethod
    @abstractmethod
    def load(path: Path, **kwargs: Any) -> T: ...  # noqa: ANN401

    @classmethod
    @abstractmethod
    def check_isinstance(cls, obj: Any) -> bool: ...  # noqa: ANN401
