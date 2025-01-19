"""Base module for all modules."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy.typing as npt

from autointent._dump_tools import Dumper
from autointent.context import Context
from autointent.context.optimization_info import Artifact
from autointent.custom_types import BaseMetadataDict


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
    ) -> dict[str, float | str]:
        """
        Calculate metric on test set and return metric value.

        :param context: Context to score
        :param split: Split to score on
        :return: Computed metrics value for the test set or error code of metrics
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
        Dumper.dump(self, Path(path))

    def load(self, path: str) -> None:
        """
        Load data from dump.

        :param path: Path to load
        """
        Dumper.load(self, Path(path))

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

    @staticmethod
    def score_metrics(params: tuple[Any, Any], metrics_dict: dict[str, Any]) -> dict[str, float | str]:
        """
        Score metrics on the test set.

        :param params: Params to score
        :param metrics_dict:
        :return:
        """
        metrics = {}
        for metric_name, metric_fn in metrics_dict.items():
            try:
                metrics[metric_name] = metric_fn(*params)
            except Exception as e:  # noqa: PERF203, BLE001
                metrics[metric_name] = str(e)
        return metrics
