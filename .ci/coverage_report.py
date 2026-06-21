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
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import coverage

logger = logging.getLogger("coverage_report")


def main() -> int:
    """Combine coverage data, emit reports, and write the GitHub step summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    cov = coverage.Coverage()
    cov.combine()  # merge the downloaded `.coverage.*` files into the data file
    cov.load()

    total = cov.report()  # full table (config `show_missing`) -> job log
    cov.xml_report()
    cov.html_report()

    logger.info("Total coverage: %.2f%%", total)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        compact = io.StringIO()
        cov.report(file=compact, show_missing=False)
        body = f"### Test coverage: {total:.2f}%\n\n```\n{compact.getvalue()}```\n"
        with Path(summary_path).open("a", encoding="utf-8") as fh:
            fh.write(body)

    return 0


if __name__ == "__main__":
    sys.exit(main())
