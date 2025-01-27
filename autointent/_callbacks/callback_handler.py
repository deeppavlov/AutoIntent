from pathlib import Path
from typing import Any

from autointent._callbacks.base import OptimizerCallback


class CallbackHandler(OptimizerCallback):
    """Internal class that just calls the list of callbacks in order."""

    callbacks: list[OptimizerCallback]

    def __init__(self, callbacks: list[type[OptimizerCallback]] | None = None) -> None:
        """Initialize the callback handler."""
        if not callbacks:
            self.callbacks = []
            return

        self.callbacks = [cb() for cb in callbacks]

    def start_run(self, run_name: str, dirpath: Path) -> None:
        """
        Start a new run.

        :param run_name: Name of the run.
        :param dirpath: Path to the directory where the logs will be saved.
        """
        self.call_events("start_run", run_name=run_name, dirpath=dirpath)

    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        """
        Start a new module.

        :param module_name: Name of the module.
        :param num: Number of the module.
        :param module_kwargs: Module parameters.
        """
        self.call_events("start_module", module_name=module_name, num=num, module_kwargs=module_kwargs)

    def log_value(self, **kwargs: dict[str, Any]) -> None:
        """
        Log data.

        :param kwargs: Data to log.
        """
        self.call_events("log_value", **kwargs)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Log metrics during training.

        :param metrics: Metrics to log.
        """
        self.call_events("log_metrics", metrics=metrics)

    def end_module(self) -> None:
        """End a module."""
        self.call_events("end_module")

    def end_run(self) -> None:
        """End a run."""
        self.call_events("end_run")

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Log final metrics.

        :param metrics: Final metrics.
        """
        self.call_events("log_final_metrics", metrics=metrics)

    def call_events(self, event: str, **kwargs: Any) -> None:  # noqa: ANN401
        for callback in self.callbacks:
            getattr(callback, event)(**kwargs)
