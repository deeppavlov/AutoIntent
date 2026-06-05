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

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


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
