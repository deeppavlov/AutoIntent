"""Decide test matrix scope and emit it to GITHUB_OUTPUT.

Full matrix runs on push to dev and on PRs labeled `full-ci`.
Otherwise (default PR commits) only ubuntu-latest + Python 3.14 runs.
"""

from __future__ import annotations

import json
import os
import sys

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


def collect_os_list(matrix: dict) -> list[str]:
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
    if event_name == "push":
        return True
    return FULL_CI_LABEL in labels


def parse_labels(raw: str) -> list[str]:
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
    event_name = os.environ.get("EVENT_NAME", "")
    labels = parse_labels(os.environ.get("LABELS_JSON", ""))

    full = is_full(event_name, labels)
    matrix = FULL_MATRIX if full else MINIMAL_MATRIX
    warm_os = collect_os_list(matrix)

    output_path = os.environ.get("GITHUB_OUTPUT")
    payload = {
        "matrix": json.dumps(matrix),
        "warm_os": json.dumps(warm_os),
        "full": "true" if full else "false",
    }
    if output_path:
        with open(output_path, "a", encoding="utf-8") as fh:
            for key, value in payload.items():
                fh.write(f"{key}={value}\n")

    print(f"event_name={event_name}", file=sys.stderr)
    print(f"labels={labels}", file=sys.stderr)
    print(f"full={full}", file=sys.stderr)
    print(f"matrix={payload['matrix']}", file=sys.stderr)
    print(f"warm_os={payload['warm_os']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
