from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from autointent import Embedder, Ranker, VectorIndex
from autointent._wrappers import BaseTorchModuleWithVocab
from autointent.schemas import TagsList

if TYPE_CHECKING:
    from pathlib import Path

ModuleSimpleAttributes = None | str | int | float | bool | list  # type: ignore[type-arg]

ModuleAttributes: TypeAlias = (
    ModuleSimpleAttributes
    | TagsList
    | npt.NDArray[np.floating]
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
