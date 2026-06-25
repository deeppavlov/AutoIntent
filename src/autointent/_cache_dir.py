"""Resolution of the base directory for autointent on-disk caches."""

from __future__ import annotations

import os
from pathlib import Path

from appdirs import user_cache_dir


def get_cache_dir() -> Path:
    """Return the base directory for autointent on-disk caches.

    Honors the ``AUTOINTENT_CACHE_DIR`` environment variable; otherwise falls back to
    ``appdirs.user_cache_dir("autointent")``. Resolved fresh on each call so tests and
    parallel workers can redirect it via the env var.

    Note:
        Currently consumed only by the embedding cache. The structured-output cache
        still uses ``user_cache_dir("autointent")`` directly and is unaffected by this
        variable.

    Returns:
        The cache base directory as a ``Path``.
    """
    override = os.environ.get("AUTOINTENT_CACHE_DIR")
    return Path(override) if override else Path(user_cache_dir("autointent"))
