# ruff: noqa: INP001
# .ci/ is a top-level script directory invoked by GitHub Actions, not a
# Python package, so no __init__.py here (mirrors .ci/warm_hf_cache.py).
"""Decide test matrix scope and emit it to ``GITHUB_OUTPUT``.

Full matrix runs on push to ``dev`` and on PRs labeled ``full-ci``.
Otherwise (default PR commits) only ubuntu-latest + Python 3.14 runs.
On ``workflow_dispatch`` (the manual coverage run) a single
ubuntu-latest + ``DISPATCH_PYTHON`` combo runs: coverage is the union of
lines exercised, so it is OS/Python-independent and one combo keeps the
manual job cheap.

Inputs come from environment variables:

* ``EVENT_NAME`` - the GitHub event name (``push`` / ``pull_request`` /
  ``workflow_dispatch``).
* ``LABELS_JSON`` - ``toJSON(github.event.pull_request.labels.*.name)``
  from the workflow; ``null`` / missing on non-PR events.
* ``DISPATCH_PYTHON`` - Python version for the ``workflow_dispatch`` combo;
  falls back to ``DISPATCH_DEFAULT_PYTHON`` when unset/empty.
* ``GITHUB_OUTPUT`` - file the runner reads to pick up step outputs.

The script writes ``matrix``, ``warm_os`` and ``full`` to that output
file and mirrors them to the log for debuggability.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

FULL_MATRIX = {
    "os": ["ubuntu-latest"],
    "python-version": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "include": [{"os": "windows-latest", "python-version": "3.10"}],
}

MINIMAL_MATRIX = {
    "os": ["ubuntu-latest"],
    "python-version": ["3.14"],
}

FULL_CI_LABEL = "full-ci"

DISPATCH_EVENT = "workflow_dispatch"
DISPATCH_DEFAULT_PYTHON = "3.12"


def dispatch_matrix(python_version: str) -> dict:
    """Return the single-combo matrix for the manual coverage run."""
    return {"os": ["ubuntu-latest"], "python-version": [python_version or DISPATCH_DEFAULT_PYTHON]}


logger = logging.getLogger("compute_matrix")


def collect_os_list(matrix: dict) -> list[str]:
    """Return the unique runner OSes referenced by ``matrix`` (base + includes)."""
    seen: list[str] = []
    for entry in matrix.get("os", []):
        if entry not in seen:
            seen.append(entry)
    for include in matrix.get("include", []):
        entry = include.get("os")
        if entry and entry not in seen:
            seen.append(entry)
    return seen


def is_full(event_name: str, labels: list[str]) -> bool:
    """Return True iff this run should fan out across the full OS/Python matrix."""
    # Any push that reaches this workflow is a push to `dev` (ci.yaml pins
    # on.push.branches: [dev]), so the branch is implied and not re-checked
    # here. If more push branches are ever added there, revisit this.
    if event_name == "push":
        return True
    return FULL_CI_LABEL in labels


def parse_labels(raw: str) -> list[str]:
    """Parse the ``LABELS_JSON`` env var into a list of label names.

    Returns an empty list when the value is missing, ``"null"`` (the
    ``toJSON`` rendering of a missing PR object), malformed, or not a
    JSON array of strings.
    """
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [item for item in decoded if isinstance(item, str)]


def main() -> int:
    """Compute matrix from env, log a summary, write outputs; return exit code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    event_name = os.environ.get("EVENT_NAME", "")
    labels = parse_labels(os.environ.get("LABELS_JSON", ""))

    if event_name == DISPATCH_EVENT:
        full = False
        matrix = dispatch_matrix(os.environ.get("DISPATCH_PYTHON", ""))
    else:
        full = is_full(event_name, labels)
        matrix = FULL_MATRIX if full else MINIMAL_MATRIX
    warm_os = collect_os_list(matrix)

    payload = {
        "matrix": json.dumps(matrix),
        "warm_os": json.dumps(warm_os),
        "full": "true" if full else "false",
    }

    logger.info("event_name=%s", event_name)
    logger.info("labels=%s", labels)
    logger.info("full=%s", full)
    logger.info("matrix=%s", payload["matrix"])
    logger.info("warm_os=%s", payload["warm_os"])

    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        logger.error("GITHUB_OUTPUT is not set; cannot emit step outputs")
        return 1
    lines = "".join(f"{key}={value}\n" for key, value in payload.items())
    with Path(output_path).open("a", encoding="utf-8") as fh:
        fh.write(lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
