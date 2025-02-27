import os
from pathlib import Path
from typing import Any

from autointent._callbacks.base import OptimizerCallback


class WandbCallback(OptimizerCallback):
    """Wandb callback for logging the optimization process to Weights & Biases (W&B).

    This callback integrates with W&B to track training runs, log metrics, and store
    configurations.

    To specify the project name, set the `WANDB_PROJECT` environment variable. If not set,
    it defaults to `autointent`.
    """

    name = "wandb"

    def __init__(self) -> None:
        """Initializes the Wandb callback.

        Ensures that `wandb` is installed before using this callback.

        Raises:
            ImportError: If `wandb` is not installed.
        """
        try:
            import wandb
        except ImportError:
            msg = "Please install wandb to use this callback. `pip install wandb`"
            raise ImportError(msg) from None

        self.wandb = wandb

    def start_run(self, run_name: str, dirpath: Path) -> None:
        """Starts a new W&B run.

        Initializes the project name and run group. The directory path argument is not
        used in this callback.

        Args:
            run_name: Name of the run (used as a W&B group).
            dirpath: Path to store logs (not utilized in W&B logging).
        """
        self.project_name = os.getenv("WANDB_PROJECT", "autointent")
        self.group = run_name
        self.dirpath = dirpath

    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        """Starts a new module within the W&B logging system.

        This initializes a W&B run with the specified module name, unique identifier,
        and configuration parameters.

        Args:
            module_name: The name of the module being logged.
            num: A numerical identifier for the module instance.
            module_kwargs: Dictionary containing module parameters.
        """
        self.wandb.init(
            project=self.project_name,
            group=self.group,
            name=f"{module_name}_{num}",
            config=module_kwargs,
        )

    def log_value(self, **kwargs: dict[str, Any]) -> None:
        """Logs scalar or textual values to W&B.

        This function logs the provided key-value pairs to W&B.

        Args:
            **kwargs: Key-value pairs of data to log.
        """
        self.wandb.log(kwargs)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Logs training metrics to W&B.

        Args:
            metrics: A dictionary containing metric names and values.
        """
        self.wandb.log(metrics)

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        """Logs final evaluation metrics to W&B.

        A new W&B run named `final_metrics` is created to store the final performance metrics.

        Args:
            metrics: A dictionary of final performance metrics.
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
        """Ends the current W&B module.

        This closes the W&B run associated with the current module.
        """
        self.wandb.finish()

    def end_run(self) -> None:
        """Ends the W&B run.

        This method is currently a placeholder and does not perform additional operations.
        """
