"""Emissions tracking functionality for monitoring energy consumption and carbon emissions."""

import json
import logging
import os

from codecarbon import EmissionsTracker as CodeCarbonTracker  # type: ignore[import-untyped]
from codecarbon.output import EmissionsData  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class EmissionsTracker:
    """Class for tracking energy consumption and carbon emissions."""

    def __init__(self, project_name: str, measure_power_secs: int = 1) -> None:
        """Initialize the emissions tracker.

        Args:
            project_name: Name of the project to track emissions for.
            measure_power_secs: How often to measure power consumption in seconds.
        """
        self._logger = logger
        self._enabled = int(os.getenv("TRACK_EMISSIONS", "0"))
        if self._enabled:
            self._logger.info("Emissions tracking is enabled via TRACK_EMISSIONS environment variable")
            self.tracker = CodeCarbonTracker(project_name=project_name, measure_power_secs=measure_power_secs)
        else:
            self.tracker = None

    def start_task(self, task_name: str) -> None:
        """Start tracking emissions for a specific task.

        Args:
            task_name: Name of the task to track emissions for.
        """
        if self._enabled:
            self.tracker.start_task(task_name)

    def stop_task(self) -> dict[str, float]:
        """Stop tracking emissions and return the emissions data.

        Returns:
            Dictionary containing emissions metrics.
        """
        if not self._enabled:
            return {}

        emissions_data = self.tracker.stop_task()
        _ = self.tracker.stop()
        return self._process_metrics(emissions_data)

    def _process_metrics(self, emissions_data: EmissionsData) -> dict[str, float]:
        """Process emissions data into metrics with the 'emissions/' prefix.

        Args:
            emissions_data: Raw emissions data from the tracker.

        Returns:
            Dictionary of processed emissions metrics with the 'emissions/' prefix.
        """
        emissions_data_dict = json.loads(emissions_data.toJSON())
        return {f"emissions/{k}": v for k, v in emissions_data_dict.items() if isinstance(v, int | float)}
