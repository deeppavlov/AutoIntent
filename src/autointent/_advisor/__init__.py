"""Pre-flight compute feasibility advisor.

Exposes a small surface used by both ``Pipeline.fit()`` (future integration) and
the ``autointent-advisor`` CLI script. See ``compute-feasibility-advisor-proposal.md``
at the repo root for the design document.
"""

from __future__ import annotations

from ._estimates import run_preflight
from ._hardware import HardwareProfile, detect_hardware
from ._report import DatasetStats, Finding, PreflightReport, RecommendationResult, ResourceEstimate, Severity
from ._workflows import inspect, load_config, recommend, stats_from_dataset

__all__ = [
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
]
