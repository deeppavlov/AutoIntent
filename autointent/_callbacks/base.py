"""Base class for reporters (W&B, TensorBoard, etc)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class OptimizerCallback(ABC):
    """Base class for reporters (W&B, TensorBoard, etc)."""

    # Implementation inspired by TrainerCallback from HuggingFace Transformers. https://github.com/huggingface/transformers/blob/91b8ab18b778ae9e2f8191866e018cd1dc7097be/src/transformers/trainer_callback.py#L260
    name: str

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def start_run(self, run_name: str, dirpath: Path, log_interval_time: float) -> None:
        """Start a new run.

        Args:
            run_name: Name of the run.
            dirpath: Path to the directory where the logs will be saved.
            log_interval_time: Sampling interval for the system monitor in seconds.
        """

    @abstractmethod
    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        """Start a new module.

        Args:
            module_name: Name of the module.
            num: Number of the module.
            module_kwargs: Module parameters.
        """

    @abstractmethod
    def log_value(self, **kwargs: dict[str, Any]) -> None:
        """Log data.

        Args:
            kwargs: Data to log.
        """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log metrics during training.

        Args:
            metrics: Metrics to log.
        """

    @abstractmethod
    def end_module(self) -> None:
        """End a module."""

    @abstractmethod
    def end_run(self) -> None:
        """End a run."""

    @abstractmethod
    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        """Log final metrics.

        Args:
            metrics: Final metrics.
        """

    def update_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Update metrics during training.

        Args:
            metrics: Metrics to update.
        """
        return metrics

    def update_final_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Update final metrics.

        Args:
            metrics: Final metrics to update.

        Returns:
            Updated final metrics.
        """
        return metrics
