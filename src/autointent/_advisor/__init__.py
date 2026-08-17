"""Pre-flight compute feasibility advisor.

Exposes a small surface used by both ``Pipeline.fit()`` (see the ``preflight=``
kwarg) and the ``autointent-advisor`` CLI script.
"""

from __future__ import annotations

from ._hardware import HardwareProfile, detect_hardware
from ._report import DatasetStats, Finding, PreflightReport, RecommendationResult, ResourceEstimate, Severity
from .runner import run_preflight
from .workflows import (
    BUNDLED_PRESETS,
    ReduceToFitError,
    inspect,
    load_config,
    recommend,
    reduce_to_fit,
    stats_from_dataset,
    stats_from_dataset_obj,
)

__all__ = [
    "BUNDLED_PRESETS",
    "DatasetStats",
    "Finding",
    "HardwareProfile",
    "PreflightReport",
    "RecommendationResult",
    "ReduceToFitError",
    "ResourceEstimate",
    "Severity",
    "detect_hardware",
    "inspect",
    "load_config",
    "recommend",
    "reduce_to_fit",
    "run_preflight",
    "stats_from_dataset",
    "stats_from_dataset_obj",
]
