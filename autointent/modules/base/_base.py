"""Base module for all modules."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, assert_never

from autointent._dump_tools import Dumper
from autointent.configs import CrossEncoderConfig, EmbedderConfig
from autointent.context import Context
from autointent.context.optimization_info import Artifact
from autointent.custom_types import ListOfGenericLabels, ListOfLabels
from autointent.exceptions import WrongClassificationError

logger = logging.getLogger(__name__)


class BaseModule(ABC):
    """Base module for all intent classification modules."""

    supports_oos: bool
    """Whether the module supports oos data"""
    supports_multilabel: bool
    """Whether the module supports multilabel classification"""
    supports_multiclass: bool
    """Whether the module supports multiclass classification"""
    name: str
    """Name of the module to reference in search space configuration."""

    @property
    def trial_name(self) -> str:
        """Name of the module for logging."""
        return self.name

    @abstractmethod
    def fit(self, *args: tuple[Any], **kwargs: dict[str, Any]) -> None:
        """Fit the model.

        Args:
            *args: Args to fit
            **kwargs: Kwargs to fit
        """

    def score(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Calculate metric on test set and return metric value.

        Args:
            context: Context to score
            metrics: Metrics to score

        Raises:
            ValueError: If unknown scheme is provided
        """
        if context.data_handler.config.scheme == "ho":
            return self.score_ho(context, metrics)
        if context.data_handler.config.scheme == "cv":
            return self.score_cv(context, metrics)
        assert_never(context.data_handler.config.scheme)

    @abstractmethod
    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]: ...

    @abstractmethod
    def score_ho(self, context: Context, metrics: list[str]) -> dict[str, float]: ...

    @abstractmethod
    def get_assets(self) -> Artifact:
        """Return useful assets that represent intermediate data into context.

        Returns:
            Artifact containing intermediate data
        """

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear cache."""

    def dump(self, path: str) -> None:
        """Dump all data needed for inference.

        Args:
            path: Path to dump
        """
        Dumper.dump(self, Path(path))

    @classmethod
    def load(
        cls,
        path: str,
        embedder_config: EmbedderConfig | None = None,
        cross_encoder_config: CrossEncoderConfig | None = None,
    ) -> Self:
        """Load data from file system.

        Args:
            path: Path to load
            embedder_config: one can override presaved settings
            cross_encoder_config: one can override presaved settings
        """
        instance = cls()
        Dumper.load(instance, Path(path), embedder_config=embedder_config, cross_encoder_config=cross_encoder_config)
        return instance

    @abstractmethod
    def predict(
        self, *args: list[str] | npt.NDArray[Any], **kwargs: dict[str, Any]
    ) -> ListOfGenericLabels | npt.NDArray[Any]:
        """Predict on the input.

        Args:
            *args: args to predict
            **kwargs: kwargs to predict
        """

    def predict_with_metadata(
        self,
        *args: list[str] | npt.NDArray[Any],
        **kwargs: dict[str, Any],
    ) -> tuple[ListOfGenericLabels | npt.NDArray[Any], list[dict[str, Any]] | None]:
        """Predict on the input with metadata.

        Args:
            *args: args to predict
            **kwargs: kwargs to predict

        Returns:
            Tuple of predictions and metadata
        """
        return self.predict(*args, **kwargs), None

    @classmethod
    @abstractmethod
    def from_context(cls, context: Context, **kwargs: dict[str, Any]) -> "BaseModule":
        """Initialize self from context.

        Args:
            context: Context to init from
            **kwargs: Additional kwargs

        Returns:
            Initialized module
        """

    @abstractmethod
    def get_implicit_initialization_params(self) -> dict[str, Any]:
        """Return default params used in ``__init__`` method.

        Some parameters of the module may be inferred using context rather from ``__init__`` method.
        But they need to be logged for reproducibility during loading from disk.

        Returns:
            Dictionary of default params
        """

    @staticmethod
    def score_metrics_ho(params: tuple[Any, Any], metrics_dict: dict[str, Any]) -> dict[str, float]:
        """Score metrics on the test set.

        Args:
            params: Params to score
            metrics_dict: Dictionary of metrics to compute
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
        """Score metrics using cross-validation.

        Args:
            metrics_dict: Dictionary of metrics to compute
            cv_iterator: Cross-validation iterator
            **fit_kwargs: Additional arguments for fit method
        """
        metrics_values: dict[str, list[float]] = {name: [] for name in metrics_dict}
        all_val_preds = []

        for train_utterances, train_labels, val_utterances, val_labels in cv_iterator:
            self.clear_cache()
            self.fit(train_utterances, train_labels, **fit_kwargs)  # type: ignore[arg-type]
            val_preds = self.predict(val_utterances)
            for name, fn in metrics_dict.items():
                metrics_values[name].append(fn(val_labels, val_preds))
            all_val_preds.append(val_preds)

        metrics = {name: float(np.mean(values_list)) for name, values_list in metrics_values.items()}
        return metrics, all_val_preds  # type: ignore[return-value]

    def _validate_multilabel(self, data_is_multilabel: bool) -> None:
        """Validate if module supports the required classification type.

        Args:
            data_is_multilabel: Whether the data is multilabel

        Raises:
            WrongClassificationError: If module doesn't support the required classification type
        """
        if data_is_multilabel and not self.supports_multilabel:
            msg = f'"{self.name}" module is incompatible with multi-label classifiction.'
            logger.error(msg)
            raise WrongClassificationError(msg)
        if not data_is_multilabel and not self.supports_multiclass:
            msg = f'"{self.name}" module is incompatible with multi-class classifiction.'
            logger.error(msg)
            raise WrongClassificationError(msg)

    def _validate_oos(self, data_contains_oos: bool, raise_error: bool = True) -> None:
        """Validate if module supports out-of-scope samples.

        Args:
            data_contains_oos: Whether data contains OOS samples
            raise_error: Whether to raise error on validation failure

        Raises:
            ValueError: If validation fails and raise_error is True
        """
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
        """Validate task specifications.

        Args:
            labels: Training labels
        """
        self._n_classes, self._multilabel, self._oos = self._get_task_specs(labels)
        self._validate_multilabel(self._multilabel)
        self._validate_oos(self._oos)

    @staticmethod
    def _get_task_specs(labels: ListOfGenericLabels) -> tuple[int, bool, bool]:
        """Infer number of classes, type of classification and whether data contains OOS samples.

        Args:
            labels: Training labels

        Returns:
            Tuple containing:
                - number of classes
                - indicator if it's a multi-label task
                - indicator if data contains oos samples
        """
        contains_oos_samples = any(label is None for label in labels)
        in_domain_label = next(lab for lab in labels if lab is not None)
        multilabel = isinstance(in_domain_label, list)
        n_classes = len(in_domain_label) if multilabel else len(set(labels).difference([None]))  # type: ignore[arg-type]
        return n_classes, multilabel, contains_oos_samples
