"""Base module for all modules."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy.typing as npt

from autointent._dump_tools import Dumper
from autointent.context import Context
from autointent.context.optimization_info import Artifact
from autointent.custom_types import ListOfGenericLabels
from autointent.exceptions import WrongClassificationError

logger = logging.getLogger(__name__)


class Module(ABC):
    """Base module."""

    supports_oos: bool
    supports_multilabel: bool
    supports_multiclass: bool
    name: str

    @abstractmethod
    def fit(self, *args: tuple[Any], **kwargs: dict[str, Any]) -> None:
        """
        Fit the model.

        :param args: Args to fit
        :param kwargs: Kwargs to fit
        """

    @abstractmethod
    def score(self, context: Context, split: Literal["validation", "test"], metrics: list[str]) -> dict[str, float]:
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
    def predict(
        self, *args: list[str] | npt.NDArray[Any], **kwargs: dict[str, Any]
    ) -> ListOfGenericLabels | npt.NDArray[Any]:
        """
        Predict on the input.

        :param args: args to predict.
        :param kwargs: kwargs to predict.
        """

    def predict_with_metadata(
        self,
        *args: list[str] | npt.NDArray[Any],
        **kwargs: dict[str, Any],
    ) -> tuple[ListOfGenericLabels | npt.NDArray[Any], list[dict[str, Any]] | None]:
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
    def score_metrics(params: tuple[Any, Any], metrics_dict: dict[str, Any]) -> dict[str, float]:
        """
        Score metrics on the test set.

        :param params: Params to score
        :param metrics_dict:
        :return:
        """
        metrics = {}
        for metric_name, metric_fn in metrics_dict.items():
            metrics[metric_name] = metric_fn(*params)
        return metrics

    def _validate_multilabel(self, data_is_multilabel: bool) -> None:
        if data_is_multilabel and not self.supports_multilabel:
            msg = f'"{self.name}" module is incompatible with multi-label classifiction.'
            logger.error(msg)
            raise WrongClassificationError(msg)
        if not data_is_multilabel and not self.supports_multiclass:
            msg = f'"{self.name}" module is incompatible with multi-class classifiction.'
            logger.error(msg)
            raise WrongClassificationError(msg)

    def _validate_oos(self, data_contains_oos: bool, raise_error: bool = True) -> None:
        if data_contains_oos != self.supports_oos:
            if self.supports_oos and not data_contains_oos:
                msg = (
                    f'"{self.name}" is designed to handle OOS samples, but your data doesn\'t '
                    "contain any of it. So, using this method puts unnecessary computational overhead."
                )
            elif not self.supports_oos and data_contains_oos:
                msg = (
                    f'"{self.name}" is NOT designed to handle OOS samples, but your data '
                    "contains it. So, using this method reduces the power of classification."
                )
            if raise_error:
                logger.error(msg)
                raise ValueError(msg)
            logger.warning(msg)

    def _validate_task(self, labels: ListOfGenericLabels) -> None:
        self._n_classes, self._multilabel, self._oos = self._get_task_specs(labels)
        self._validate_multilabel(self._multilabel)
        self._validate_oos(self._oos)

    @staticmethod
    def _get_task_specs(labels: ListOfGenericLabels) -> tuple[int, bool, bool]:
        """
        Infer number of classes, type of classification and whether data contains OOS samples.

        :param scores: training scores
        :param labels: training labels
        :return: number of classes, indicator if it's a multi-label task,
                    indicator if data contains oos samples
        """
        contains_oos_samples = any(label is None for label in labels)
        in_domain_label = next(lab for lab in labels if lab is not None)
        multilabel = isinstance(in_domain_label, list)
        n_classes = len(labels[0]) if multilabel else len(set(labels).difference([None]))  # type: ignore[arg-type]
        return n_classes, multilabel, contains_oos_samples
