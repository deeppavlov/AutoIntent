from pathlib import Path
from typing import Any

from autointent._callbacks.base import OptimizerCallback


class CallbackHandler(OptimizerCallback):
    """Internal class that just calls the list of callbacks in order."""

    callbacks: list[OptimizerCallback]

    def __init__(self, callbacks: list[type[OptimizerCallback]] | None = None) -> None:
        """Initialize the callback handler.

        Args:
            callbacks: List of callback classes.
        """
        if not callbacks:
            self.callbacks = []
            return

        self.callbacks = [cb() for cb in callbacks]

    def start_run(self, run_name: str, dirpath: Path, log_interval_time: float) -> None:
        """Start a new run.

        Args:
            run_name: Name of the run.
            dirpath: Path to the directory where the logs will be saved.
            log_interval_time: Sampling interval for the system monitor in seconds.
        """
        self.call_events("start_run", run_name=run_name, dirpath=dirpath, log_interval_time=log_interval_time)

    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        """Start a new module.

        Args:
            module_name: Name of the module.
            num: Number of the module.
            module_kwargs: Module parameters.
        """
        self.call_events("start_module", module_name=module_name, num=num, module_kwargs=module_kwargs)

    def log_value(self, **kwargs: dict[str, Any]) -> None:
        """Log data.

        Args:
            kwargs: Data to log.
        """
        self.call_events("log_value", **kwargs)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log metrics during training.

        Args:
            metrics: Metrics to log.
        """
        self.call_events("log_metrics", metrics=metrics)

    def end_module(self) -> None:
        """End a module."""
        self.call_events("end_module")

    def end_run(self) -> None:
        """End a run."""
        self.call_events("end_run")

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        """Log final metrics.

        Args:
            metrics: Final metrics.
        """
        self.call_events("log_final_metrics", metrics=metrics)

    def call_events(self, event: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Call events for all callbacks.

        Args:
            event: Event name.
            kwargs: Event parameters.
        """
        for callback in self.callbacks:
            getattr(callback, event)(**kwargs)

    def update_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Update metrics during training.

        Args:
            metrics: Metrics to update.

        Returns:
            Updated metrics.
        """
        for callback in self.callbacks:
            if hasattr(callback, "update_metrics"):
                metrics = callback.update_metrics(metrics)
        return metrics

    def update_final_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Update final metrics.

        Args:
            metrics: Final metrics to update.

        Returns:
            Updated final metrics.
        """
        for callback in self.callbacks:
            if hasattr(callback, "update_final_metrics"):
                metrics = callback.update_final_metrics(metrics)
        return metrics
