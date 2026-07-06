"""Pre-flight compute feasibility advisor.

Exposes a small surface used by both ``Pipeline.fit()`` (see the ``preflight=``
kwarg) and the ``autointent-advisor`` CLI script. See
``compute-feasibility-advisor-proposal.md`` at the repo root for the design.
"""

from __future__ import annotations

from ._hardware import HardwareProfile, detect_hardware
from ._report import DatasetStats, Finding, PreflightReport, RecommendationResult, ResourceEstimate, Severity
from .runner import run_preflight
from .workflows import (
    BUNDLED_PRESETS,
    inspect,
    load_config,
    recommend,
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
    "ResourceEstimate",
    "Severity",
    "detect_hardware",
    "inspect",
    "load_config",
    "recommend",
    "run_preflight",
    "stats_from_dataset",
    "stats_from_dataset_obj",
]
