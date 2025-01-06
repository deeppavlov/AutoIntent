from pathlib import Path
from typing import Any

import numpy.typing as npt
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.base import BaseEstimator

from autointent import Embedder
from autointent.context.vector_index_client import VectorIndex


def dump_constants(obj: dict[str, str | int | float | bool | list], path: Path) -> None:
    """Dump dictionary to filestystem."""


def dump_arrays(obj: dict[str, npt.NDArray[Any]], path: Path) -> None:
    """Dump dictionary to filestystem."""


def dump_embedders(obj: dict[str, Embedder], path: Path) -> None:
    """Dump dictionary to filestystem."""


def dump_indexes(obj: dict[str, VectorIndex], path: Path) -> None:
    """Dump dictionary to filestystem."""


def dump_estimators(obj: dict[str, BaseEstimator], path: Path) -> None:
    """Dump dictionary to filestystem."""


def dump_sentence_transformers(obj: dict[str, SentenceTransformer], path: Path) -> None:
    """Dump dictionary to filestystem."""


def dump_cross_encoders(obj: dict[str, CrossEncoder], path: Path) -> None:
    """Dump dictionary to filestystem."""
