"""Pre-flight compute feasibility advisor.

Exposes a small surface used by both ``Pipeline.fit()`` (future integration) and
the ``autointent-advisor`` CLI script. See ``compute-feasibility-advisor-proposal.md``
at the repo root for the design document.
"""

from __future__ import annotations

from ._hardware import HardwareProfile, detect_hardware
from ._report import DatasetStats, Finding, PreflightReport, ResourceEstimate, Severity
from ._estimates import run_preflight

__all__ = [
    "DatasetStats",
    "Finding",
    "HardwareProfile",
    "PreflightReport",
    "ResourceEstimate",
    "Severity",
    "detect_hardware",
    "run_preflight",
]
