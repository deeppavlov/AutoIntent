# ruff: noqa: INP001
# .ci/ is a top-level script directory invoked by GitHub Actions, not a
# Python package. The tests/ci/conftest.py shim adds it to sys.path for
# import-by-name, so we don't want an __init__.py here.
"""Pre-populate the HuggingFace cache for the autointent CI test suite.

Reads ``.ci/hf-prewarm.yaml`` and ensures every listed model / dataset is
present in ``~/.cache/huggingface`` at the pinned revision. The CI workflow
runs this in a dedicated job before any test job so tests find every HF
resource on disk and never make HF API calls themselves.

Entries in the YAML are bare repo IDs. SHAs are resolved against
``autointent.configs._pinned_revisions.DEFAULT_REVISIONS`` at runtime
via a ``sys.path`` shim (see the import block below). Pinning the
revision lets the fast path (``snapshot_download`` with
``local_files_only=True``) decide "cache is complete" without calling
the HF API, which is what avoids the 1000-req/5-min rate limit on
cold-cache CI runs.

This module is also imported by ``tests/ci/test_warm_hf_cache.py``, so
keep top-level imports cheap and side-effect-free.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

# Load DEFAULT_REVISIONS from the autointent leaf module WITHOUT going
# through the package's import machinery. A plain
# `from autointent.configs._pinned_revisions import ...` would execute
# autointent/__init__.py first, which imports pydantic/numpy/etc and
# crashes in the warm-cache job's slim env. importlib.util loads the
# file directly by path, bypassing the parent package entirely.
_LEAF = Path(__file__).resolve().parent.parent / "src" / "autointent" / "configs" / "_pinned_revisions.py"
_spec = importlib.util.spec_from_file_location("_pinned_revisions", _LEAF)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DEFAULT_REVISIONS: dict[str, str] = _mod.DEFAULT_REVISIONS

logger = logging.getLogger("warm_hf_cache")

_RETRY_DELAYS = (60, 120, 180)

Outcome = Literal["cached", "downloaded", "failed"]
RepoType = Literal["model", "dataset"]


class ConfigError(ValueError):
    """Raised when ``.ci/hf-prewarm.yaml`` is malformed."""


@dataclass(frozen=True)
class Entry:
    """One HF resource to prewarm."""

    repo_type: RepoType
    repo_id: str
    revision: str


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
        default=Path(".ci/hf-prewarm-linux.yaml"),
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
        entries = _load_config(args.config)
    except ConfigError:
        logger.exception("Config error")
        return 2

    counts: dict[Outcome, int] = {"cached": 0, "downloaded": 0, "failed": 0}
    for entry in entries:
        counts[_prewarm_entry(entry)] += 1

    logger.info(
        "Summary: %d cached, %d downloaded, %d failed",
        counts["cached"],
        counts["downloaded"],
        counts["failed"],
    )
    if args.strict and counts["failed"]:
        return 1
    return 0


def _resolve_entry(repo_id: str, repo_type: RepoType) -> Entry:
    """Resolve a repo ID to a pinned Entry via DEFAULT_REVISIONS.

    Raises:
        ConfigError: if ``repo_id`` is not pinned in DEFAULT_REVISIONS.
            DEFAULT_REVISIONS covers models only today, so any
            ``repo_type="dataset"`` entry will raise here until the dict
            is extended; that's intentional (a dataset in the YAML must
            not silently regress to an unpinned download).
    """
    if repo_id not in DEFAULT_REVISIONS:
        msg = (
            f"{repo_id!r} ({repo_type}) not in DEFAULT_REVISIONS. Add a pin "
            "to src/autointent/configs/_pinned_revisions.py before listing "
            "the repo in .ci/hf-prewarm-*.yaml."
        )
        raise ConfigError(msg)
    return Entry(repo_type=repo_type, repo_id=repo_id, revision=DEFAULT_REVISIONS[repo_id])


def _load_config(path: Path) -> list[Entry]:
    """Load ``hf-prewarm.yaml`` and return a flat list of entries.

    Entries in the YAML are bare repo IDs (no ``@sha`` suffix); SHAs are
    looked up from DEFAULT_REVISIONS in _resolve_entry.

    Raises:
        ConfigError: on unknown top-level keys, or on a repo_id that's
            not pinned in DEFAULT_REVISIONS.
    """
    data = yaml.safe_load(path.read_text()) or {}
    known = {"models", "datasets"}
    unknown = set(data) - known
    if unknown:
        msg = f"Unknown top-level keys in {path}: {sorted(unknown)}"
        raise ConfigError(msg)
    entries: list[Entry] = []
    for key, repo_type in (("models", "model"), ("datasets", "dataset")):
        entries.extend(_resolve_entry(repo_id, repo_type) for repo_id in data.get(key, []) or [])
    return entries


def _populate_datasets_cache(repo_id: str) -> None:
    """Populate the ``datasets`` library cache for ``repo_id``.

    ``huggingface_hub.snapshot_download`` puts raw repo files under
    ``~/.cache/huggingface/hub/datasets--<repo>``. ``datasets.load_dataset``
    looks in a completely different location (``~/.cache/huggingface/datasets/``,
    plus its own metadata index), and won't find a dataset just because the
    Hub snapshot is on disk. We invoke ``load_dataset`` for every config of
    the dataset here so the datasets cache is populated too — that's what
    lets ``autointent.Dataset.from_hub`` work in HF_HUB_OFFLINE mode.
    """
    from datasets import get_dataset_config_names, load_dataset

    try:
        configs = get_dataset_config_names(repo_id) or ["default"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s - get_dataset_config_names failed (%s); falling back to 'default'", repo_id, exc)
        configs = ["default"]
    for config in configs:
        load_dataset(repo_id, config, download_mode="reuse_dataset_if_exists")


def _prewarm_entry(entry: Entry) -> Outcome:
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
        if entry.repo_type == "dataset":
            _populate_datasets_cache(entry.repo_id)
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
                logger.error("%s - giving up after %d attempts", label, attempt)  # noqa: TRY400
                return "failed"
            logger.info("%s - sleeping %ds before retry", label, delay)
            time.sleep(delay)
        else:
            if entry.repo_type == "dataset":
                _populate_datasets_cache(entry.repo_id)
            logger.info("%s - downloaded", label)
            return "downloaded"
    return "failed"


if __name__ == "__main__":
    sys.exit(main())
