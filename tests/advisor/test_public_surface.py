"""Locks the public surface of ``autointent.advisor``.

The advisor is marked experimental, but "experimental" is not a licence for the
surface to drift silently. This test is the tripwire: adding or removing a
public name is a deliberate act that updates this list.
"""

from __future__ import annotations

from autointent import advisor

EXPECTED_SURFACE = {
    # functions
    "dataset_stats",
    "detect_hardware",
    "estimate",
    "recommend",
    "reduce_to_fit",
    "run_preflight",
    # types
    "DatasetStats",
    "Finding",
    "HardwareProfile",
    "PreflightError",
    "PreflightReport",
    "RecommendationResult",
    "ReduceToFitError",
    "ResourceEstimate",
    "Severity",
}


def test_all_matches_expected_surface() -> None:
    assert set(advisor.__all__) == EXPECTED_SURFACE


def test_every_exported_name_resolves() -> None:
    missing = [name for name in advisor.__all__ if not hasattr(advisor, name)]
    assert missing == []


def test_no_stdlib_shadowing_names() -> None:
    """``inspect`` was exported previously and shadows the stdlib module."""
    assert "inspect" not in advisor.__all__


def test_package_documents_experimental_status() -> None:
    assert advisor.__doc__ is not None
    assert "experimental" in advisor.__doc__.lower()
