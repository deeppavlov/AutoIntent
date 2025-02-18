import os
from pathlib import Path
from typing import Any

from autointent._callbacks.base import OptimizerCallback


class WandbCallback(OptimizerCallback):
    """
    Wandb callback.

    This callback logs the optimization process to W&B.
    To specify the project name, set the `WANDB_PROJECT` environment variable. Default is `autointent`.
    """

    name = "wandb"

    def __init__(self) -> None:
        """Initialize the callback."""
        try:
            import wandb
        except ImportError:
            msg = "Please install wandb to use this callback. `pip install wandb`"
            raise ImportError(msg) from None

        self.wandb = wandb

    def start_run(self, run_name: str, dirpath: Path) -> None:
        """
        Start a new run.

        :param run_name: Name of the run.
        :param dirpath: Path to the directory where the logs will be saved. (Not used for this callback)
        """
        self.project_name = os.getenv("WANDB_PROJECT", "autointent")
        self.group = run_name
        self.dirpath = dirpath

    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        """
        Start a new module.

        :param module_name: Name of the module.
        :param num: Number of the module.
        :param module_kwargs: Module parameters.
        """
        self.wandb.init(
            project=self.project_name,
            group=self.group,
            name=f"{module_name}_{num}",
            config=module_kwargs,
        )

    def log_value(self, **kwargs: dict[str, Any]) -> None:
        """
        Log data.

        :param kwargs: Data to log.
        """
        self.wandb.log(kwargs)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Log metrics during training.

        :param metrics: Metrics to log.
        """
        self.wandb.log(metrics)

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Log final metrics.

        :param metrics: Final metrics.
        """
        self.wandb.init(
            project=self.project_name,
            group=self.group,
            name="final_metrics",
            config=metrics,
        )

        self.wandb.log(metrics.get("pipeline_metrics", {}))
        self.wandb.finish()

    def end_module(self) -> None:
        """End a module."""
        self.wandb.finish()

    def end_run(self) -> None:
        pass
