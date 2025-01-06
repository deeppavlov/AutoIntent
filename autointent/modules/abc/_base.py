"""Base module for all modules."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.base import BaseEstimator

from autointent import Embedder
from autointent._dump_tools import (
    dump_arrays,
    dump_constants,
    dump_cross_encoders,
    dump_embedders,
    dump_estimators,
    dump_indexes,
    dump_sentence_transformers,
)
from autointent.context import Context
from autointent.context.optimization_info import Artifact
from autointent.context.vector_index_client import VectorIndex
from autointent.custom_types import BaseMetadataDict
from autointent.metrics import METRIC_FN


class Module(ABC):
    """Base module."""

    name: str

    metadata_dict_name: str = "metadata.json"
    metadata: BaseMetadataDict

    @abstractmethod
    def fit(self, *args: tuple[Any], **kwargs: dict[str, Any]) -> None:
        """
        Fit the model.

        :param args: Args to fit
        :param kwargs: Kwargs to fit
        """

    @abstractmethod
    def score(
        self,
        context: Context,
        split: Literal["validation", "test"],
        metric_fn: METRIC_FN,
    ) -> float:
        """
        Calculate metric on test set and return metric value.

        :param context: Context to score
        :param split: Split to score on
        :param metric_fn: Metric function
        """

    @abstractmethod
    def get_assets(self) -> Artifact:
        """Return useful assets that represent intermediate data into context."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear cache."""

    def dump(self, path: str) -> None:
        """
        Dump all data needed for inference.

        :param path: Path to dump
        """
        constants = {}
        arrays = {}
        embedders = {}
        indexes = {}
        estimators = {}
        sentence_transformers = {}
        cross_encoders = {}

        for key, val in vars(self).items():
            if isinstance(val, str | bool | int | float | list):
                constants[key] = val
            elif isinstance(val, np.ndarray):
                arrays[key] = val
            elif isinstance(val, Embedder):
                embedders[key] = val
            elif isinstance(val, VectorIndex):
                indexes[key] = val
            elif isinstance(val, BaseEstimator):
                estimators[key] = val
            elif isinstance(val, SentenceTransformer):
                sentence_transformers[key] = val
            elif isinstance(val, CrossEncoder):
                cross_encoders[key] = val

        dump_constants(constants, Path(path))
        dump_arrays(arrays, Path(path))
        dump_embedders(embedders, Path(path))
        dump_indexes(indexes, Path(path))
        dump_estimators(estimators, Path(path))
        dump_sentence_transformers(sentence_transformers, Path(path))
        dump_cross_encoders(cross_encoders, Path(path))

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load data from dump.

        :param path: Path to load
        """

    @abstractmethod
    def predict(self, *args: list[str] | npt.NDArray[Any], **kwargs: dict[str, Any]) -> npt.NDArray[Any]:
        """
        Predict on the input.

        :param args: args to predict.
        :param kwargs: kwargs to predict.
        """

    def predict_with_metadata(
        self,
        *args: list[str] | npt.NDArray[Any],
        **kwargs: dict[str, Any],
    ) -> tuple[npt.NDArray[Any], list[dict[str, Any]] | None]:
        """
        Predict on the input with metadata.

        :param args: args to predict.
        :param kwargs: kwargs to predict.
        """
        return self.predict(*args, **kwargs), None

    @classmethod
    @abstractmethod
    def from_context(cls, context: Context, **kwargs: dict[str, Any]) -> "Module":
        """
        Initialize self from context.

        :param context: Context to init from.
        :param kwargs: Additional kwargs.
        """

    def get_embedder_name(self) -> str | None:
        """Experimental method."""
        return None
