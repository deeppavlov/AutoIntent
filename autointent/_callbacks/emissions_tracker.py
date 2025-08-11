"""Emissions tracking functionality for monitoring energy consumption and carbon emissions."""

import json
import logging
from pathlib import Path
from typing import Any

from autointent._callbacks import OptimizerCallback

logger = logging.getLogger(__name__)


class EmissionsTrackerCallback(OptimizerCallback):
    """Class for tracking energy consumption and carbon emissions."""

    name = "codecarbon"

    current_module_name: str | None = None

    def __init__(self) -> None:
        """Initialize the emission tracker."""
        try:
            from codecarbon import EmissionsTracker
        except ImportError as e:
            msg = (
                "EmissionsTrackerCallback requires the codecarbon package to be installed. "
                "Please install it with `pip install autointent[codecarbon]`."
            )
            raise ImportError(msg) from e
        self.emission_tracker = EmissionsTracker

    def start_run(self, run_name: str, dirpath: Path, log_interval_time: float) -> None:  # noqa: ARG002
        """Start tracking emissions for the entire run.

        Args:
            run_name: Name of the run.
            dirpath: Path to the directory where the logs will be saved.
            log_interval_time: Sampling interval for the system monitor in seconds.
        """
        self.tracker = self.emission_tracker(project_name=run_name, measure_power_secs=log_interval_time)

        self.tracker.start()
        self.current_module_name = None

    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:  # noqa: ARG002
        """Start tracking emissions for a specific task.

        Args:
            module_name: Name of the task to track emissions for.
            num: Number of the module.
            module_kwargs: Module parameters.
        """
        self.current_module_name = f"{module_name}_{num}"
        self.tracker.start_task(self.current_module_name)

    def update_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Stop tracking emissions and return the emissions data.

        Returns:
            Dictionary containing emissions metrics.
        """
        emissions_data = self.tracker.stop_task(self.current_module_name)
        emissions_data_json = json.loads(emissions_data.toJSON())
        emissions_data_dict = {
            f"emissions/{k}": v for k, v in emissions_data_json.items() if isinstance(v, int | float)
        }
        return emissions_data_dict | metrics

    def update_final_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Update final metrics with emissions data.

        Args:
            metrics: Final metrics to update.

        Returns:
            Updated metrics including emissions data.
        """
        _ = self.tracker.stop()
        emissions_data_json = json.loads(self.tracker.final_emissions_data.toJSON())
        emissions_data_dict = {
            f"emissions/{k}": v for k, v in emissions_data_json.items() if isinstance(v, int | float)
        }
        return {"emissions": emissions_data_dict} | metrics

    def log_value(self, **kwargs: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        pass

    def end_module(self) -> None:
        pass

    def end_run(self) -> None:
        pass

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        pass
