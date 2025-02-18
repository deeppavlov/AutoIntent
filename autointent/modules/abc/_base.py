"""Base module for all modules."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent._dump_tools import Dumper
from autointent.context import Context
from autointent.context.optimization_info import Artifact
from autointent.custom_types import ListOfGenericLabels, ListOfLabels
from autointent.exceptions import WrongClassificationError

logger = logging.getLogger(__name__)


class BaseModule(ABC):
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

    def score(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """
        Calculate metric on test set and return metric value.

        :param context: Context to score
        :param split: Split to score on
        :return: Computed metrics value for the test set or error code of metrics
        """
        if context.data_handler.config.scheme == "ho":
            return self.score_ho(context, metrics)
        if context.data_handler.config.scheme == "cv":
            return self.score_cv(context, metrics)
        msg = "Something's wrong with validation schemas"
        raise RuntimeError(msg)

    @abstractmethod
    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]: ...

    @abstractmethod
    def score_ho(self, context: Context, metrics: list[str]) -> dict[str, float]: ...

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
    def from_context(cls, context: Context, **kwargs: dict[str, Any]) -> "BaseModule":
        """
        Initialize self from context.

        :param context: Context to init from.
        :param kwargs: Additional kwargs.
        """

    def get_embedder_config(self) -> dict[str, Any] | None:
        """
        Get the config of the embedder.

        :return: Embedder config.
        """
        return None

    @staticmethod
    def score_metrics_ho(params: tuple[Any, Any], metrics_dict: dict[str, Any]) -> dict[str, float]:
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

    def score_metrics_cv(  # type: ignore[no-untyped-def]
        self,
        metrics_dict: dict[str, Any],
        cv_iterator: Iterable[tuple[list[str], ListOfLabels, list[str], ListOfLabels]],
        **fit_kwargs,  # noqa: ANN003
    ) -> tuple[dict[str, float], list[ListOfGenericLabels] | list[npt.NDArray[Any]]]:
        metrics_values: dict[str, list[float]] = {name: [] for name in metrics_dict}
        all_val_preds = []

        for train_utterances, train_labels, val_utterances, val_labels in cv_iterator:
            self.fit(train_utterances, train_labels, **fit_kwargs)  # type: ignore[arg-type]
            val_preds = self.predict(val_utterances)
            for name, fn in metrics_dict.items():
                metrics_values[name].append(fn(val_labels, val_preds))
            all_val_preds.append(val_preds)

        metrics = {name: float(np.mean(values_list)) for name, values_list in metrics_values.items()}
        return metrics, all_val_preds  # type: ignore[return-value]

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
        n_classes = len(in_domain_label) if multilabel else len(set(labels).difference([None]))  # type: ignore[arg-type]
        return n_classes, multilabel, contains_oos_samples
