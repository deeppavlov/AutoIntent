# ruff: noqa: INP001
# .ci/ is a top-level script directory invoked by GitHub Actions, not a
# Python package. The tests/ci/conftest.py shim adds it to sys.path for
# import-by-name, so we don't want an __init__.py here.
"""Pre-populate the HuggingFace cache for the autointent CI test suite.

Reads ``.ci/hf-prewarm.yaml`` and ensures every listed model / dataset is
present in ``~/.cache/huggingface`` at the pinned revision. The CI workflow
runs this in a dedicated job before any test job so tests find every HF
resource on disk and never make HF API calls themselves.

Entries are ``"<repo_id>@<sha>"`` where ``sha`` is a 40-char hex commit
hash. Pinning to a SHA lets the fast path (``snapshot_download`` with
``local_files_only=True``) decide "cache is complete" without calling the
HF API, which is what avoids the 1000-req/5-min rate limit on cold-cache
CI runs.

This module is also imported by ``tests/ci/test_warm_hf_cache.py``, so
keep top-level imports cheap and side-effect-free.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

logger = logging.getLogger("warm_hf_cache")

# Backoff sized for the HF rate limit window (per 5 min for authenticated
# users). Shorter waits usually hit the same throttle bucket and burn
# through retries; the totals here ride out two full windows in the worst
# case (60 + 120 + 180 = 360s).
_RETRY_DELAYS = (60, 120, 180)

Outcome = Literal["cached", "downloaded", "failed"]


class ConfigError(ValueError):
    """Raised when ``.ci/hf-prewarm.yaml`` is malformed."""


@dataclass(frozen=True)
class Entry:
    """One HF resource to prewarm."""

    repo_type: str  # "model" or "dataset"
    repo_id: str
    revision: str


def parse_entry(text: str) -> tuple[str, str]:
    """Split ``"<repo>@<sha>"`` into ``(repo, sha)`` and validate the SHA shape.

    Raises:
        ConfigError: if no ``@`` is present, or if the revision is not a
            40-char lowercase hex string.
    """
    if "@" not in text:
        msg = f"Entry {text!r} must be pinned to a SHA (use 'repo@<40-char hex>')"
        raise ConfigError(msg)
    repo, _, rev = text.partition("@")
    if not _SHA_RE.fullmatch(rev):
        msg = f"Entry {text!r}: revision {rev!r} is not a 40-char hex SHA"
        raise ConfigError(msg)
    return repo, rev


def load_config(path: Path) -> list[Entry]:
    """Load ``hf-prewarm.yaml`` and return a flat list of entries.

    Raises:
        ConfigError: on unknown top-level keys or malformed entries.
    """
    data = yaml.safe_load(path.read_text()) or {}
    known = {"models", "datasets"}
    unknown = set(data) - known
    if unknown:
        msg = f"Unknown top-level keys in {path}: {sorted(unknown)}"
        raise ConfigError(msg)
    entries: list[Entry] = []
    for key, repo_type in (("models", "model"), ("datasets", "dataset")):
        for raw in data.get(key, []) or []:
            repo, rev = parse_entry(raw)
            entries.append(Entry(repo_type=repo_type, repo_id=repo, revision=rev))
    return entries


def prewarm_entry(entry: Entry) -> Outcome:
    """Ensure ``entry`` is fully present in the local HF cache.

    Returns ``"cached"`` if every file was already on disk (no HF API
    contact at all), ``"downloaded"`` after a successful network pull, or
    ``"failed"`` if all retries were exhausted.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    label = f"{entry.repo_type}:{entry.repo_id}@{entry.revision[:8]}"

    # Fast path: every file already on disk -> no API call.
    try:
        snapshot_download(
            repo_id=entry.repo_id,
            revision=entry.revision,
            repo_type=entry.repo_type,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        pass  # Fall through to network download.
    else:
        logger.info("%s - cached", label)
        return "cached"

    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            snapshot_download(
                repo_id=entry.repo_id,
                revision=entry.revision,
                repo_type=entry.repo_type,
            )
        except (HfHubHTTPError, OSError) as exc:  # noqa: PERF203 - retry-with-backoff is the whole point of this loop
            logger.warning("%s - attempt %d failed (%s)", label, attempt, exc)
            if delay is None:
                # Don't use logger.exception here: the warning above already
                # captured the exception; this is just the give-up summary.
                logger.error("%s - giving up after %d attempts", label, attempt)  # noqa: TRY400
                return "failed"
            logger.info("%s - sleeping %ds before retry", label, delay)
            time.sleep(delay)
        else:
            logger.info("%s - downloaded", label)
            return "downloaded"
    return "failed"  # unreachable, keeps the type checker happy


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns 0 on success in non-strict mode (even when individual entries
    failed - best-effort by design). Returns non-zero only when ``--strict``
    is set and at least one entry failed, or when the config itself is
    malformed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".ci/hf-prewarm.yaml"),
        help="Path to the prewarm config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any entry failed to prewarm.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        entries = load_config(args.config)
    except ConfigError:
        logger.exception("Config error")
        return 2

    counts: dict[Outcome, int] = {"cached": 0, "downloaded": 0, "failed": 0}
    for entry in entries:
        counts[prewarm_entry(entry)] += 1

    logger.info(
        "Summary: %d cached, %d downloaded, %d failed",
        counts["cached"],
        counts["downloaded"],
        counts["failed"],
    )
    if args.strict and counts["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
