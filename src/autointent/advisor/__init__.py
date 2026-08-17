"""Pre-flight compute feasibility advisor.

**Experimental.** This subpackage estimates VRAM, RAM, disk, and wall-time for a
search space before any training starts. Estimates are heuristic and calibrated
against a limited hardware sample, so treat them as guidance rather than
guarantees — see the accuracy caveats in the ``advisor`` page of the docs. The
public surface may change in a minor release.

Two ways in: the ``autointent-advisor`` console script, and the functions below.
``Pipeline.fit(preflight=...)`` wires the same machinery into a fit, opt-in.
"""

from __future__ import annotations

from ._errors import PreflightError
from ._hardware import HardwareProfile, detect_hardware
from ._report import DatasetStats, Finding, PreflightReport, RecommendationResult, ResourceEstimate, Severity
from ._runner import run_preflight
from ._workflows import ReduceToFitError, dataset_stats, estimate, recommend, reduce_to_fit

__all__ = [
    "DatasetStats",
    "Finding",
    "HardwareProfile",
    "PreflightError",
    "PreflightReport",
    "RecommendationResult",
    "ReduceToFitError",
    "ResourceEstimate",
    "Severity",
    "dataset_stats",
    "detect_hardware",
    "estimate",
    "recommend",
    "reduce_to_fit",
    "run_preflight",
]
