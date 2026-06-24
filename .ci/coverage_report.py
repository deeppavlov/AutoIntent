# ruff: noqa: INP001
# .ci/ is a top-level script directory invoked by GitHub Actions, not a
# Python package, so no __init__.py here (mirrors .ci/compute_matrix.py).
"""Combine per-group coverage data files and publish a combined report.

The coverage run (``ci.yaml`` dispatched with ``coverage=true``) has every test
job upload a ``.coverage.<group>`` data file as an artifact. This script runs in
the ``coverage-report`` job after those artifacts are downloaded into the
working directory. It combines them into one dataset, prints the table to the
job log, writes ``coverage.xml`` + an HTML report (uploaded as an artifact), and
appends a compact summary to ``GITHUB_STEP_SUMMARY`` so the total is visible in
the Actions UI without opening the logs.

Run via ``uv run --no-project --with 'coverage[toml]' python .ci/coverage_report.py``;
coverage settings are read from ``[tool.coverage.*]`` in ``pyproject.toml``.

The script also enforces a regression floor on the *combined* total
(``MIN_TOTAL_COVERAGE``): a dispatch whose total drops below it fails this job.
The threshold lives here rather than in ``[tool.coverage.report] fail_under``
on purpose — pytest-cov reads that key, and each per-job ``--cov`` run measures
only a slice of the package, so a config-level ``fail_under`` would fail every
partial run. Enforcing here gates the combined total only.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import coverage

logger = logging.getLogger("coverage_report")

# Minimum acceptable combined coverage (%). Bump this as coverage improves to
# ratchet the floor up; keep it a few points below the current total so normal
# churn doesn't trip it.
MIN_TOTAL_COVERAGE = 85.0


def main() -> int:
    """Combine coverage data, emit reports, write the GitHub step summary, gate the total."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    cov = coverage.Coverage()
    cov.combine()  # merge the downloaded `.coverage.*` files into the data file
    cov.load()

    total = cov.report()  # full table (config `show_missing`) -> job log
    cov.xml_report()
    cov.html_report()

    logger.info("Total coverage: %.2f%%", total)
    passed = total >= MIN_TOTAL_COVERAGE

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        compact = io.StringIO()
        cov.report(file=compact, show_missing=False)
        gate_icon = "✅" if passed else "❌"
        gate_line = f"{gate_icon} Gate: {total:.2f}% vs {MIN_TOTAL_COVERAGE:.2f}% minimum\n"
        body = f"### Test coverage: {total:.2f}%\n\n{gate_line}\n```\n{compact.getvalue()}```\n"
        with Path(summary_path).open("a", encoding="utf-8") as fh:
            fh.write(body)

    if not passed:
        logger.error("Coverage %.2f%% is below the required minimum of %.2f%%", total, MIN_TOTAL_COVERAGE)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
