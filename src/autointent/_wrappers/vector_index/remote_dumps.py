"""Engine-agnostic deletion of dumps that reference remote cluster state (issue #343).

A backend whose ``dump()`` leaves data in an external engine marks the dump with a
``remote_manifest.json``. Deleting such a dump with a bare ``shutil.rmtree`` strands
the referenced cluster index; :func:`remove_module_dump` deletes both.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from .base_backend import MANIFEST_FILENAME
from .opensearch import OpenSearchBackend

logger = logging.getLogger(__name__)


def _delete_remote_index(manifest_path: Path) -> None:
    """Best-effort deletion of the cluster index one manifest references."""
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            engine = json.load(file).get("engine")
        if engine == "opensearch":
            OpenSearchBackend.delete_dumped_generation(manifest_path.parent)
        else:
            logger.warning("unknown remote dump engine %r in %s; skipping cluster cleanup", engine, manifest_path)
    except Exception:
        logger.exception("failed to delete remote index referenced by %s", manifest_path)


def remove_module_dump(path: str | Path) -> None:
    """Remove a module (or whole pipeline) dump directory together with its remote state.

    Scans the tree for ``remote_manifest.json`` files, deletes each referenced cluster
    index (any future remote backend that drops a manifest inherits this cleanup), then
    removes the directory. Cleanup is best-effort: cluster errors are logged, not raised,
    matching the ``shutil.rmtree(..., ignore_errors=True)`` this replaces in the
    optimization loop. Dumps without manifests (Faiss, pre-#343) are simply removed.
    """
    path = Path(path)
    for manifest_path in path.rglob(MANIFEST_FILENAME):
        _delete_remote_index(manifest_path)
    shutil.rmtree(path, ignore_errors=True)  # workaround for windows
