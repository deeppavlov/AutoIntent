"""Validate optional-extra dependencies from installed package metadata.

The :func:`require` guard checks that every dependency of an ``autointent`` extra
is installed and version-satisfied. It reads the metadata that the build baked into
the installed distribution (via :mod:`importlib.metadata`) rather than the source
``pyproject.toml``, which is not shipped in the wheel. Nested extras are resolved
recursively, so e.g. the ``transformers`` extra (``transformers[torch]``)
transitively requires ``accelerate`` and that is checked too.
"""

from __future__ import annotations

from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_DIST = "autointent"


def _check(req: Requirement) -> str | None:
    """Check a single requirement against the installed environment.

    Args:
        req: The parsed requirement to validate.

    Returns:
        A human-readable problem description if the distribution is missing or its
        installed version does not satisfy ``req.specifier``; ``None`` otherwise.
    """
    try:
        installed = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        return f"{req.name}{req.specifier} (not installed)"
    if req.specifier and not req.specifier.contains(installed, prereleases=True):
        return f"{req.name}{req.specifier} (installed: {installed})"
    return None


def _iter_extra_reqs(dist: str, extra: str) -> list[Requirement]:
    """Return the requirements of ``dist`` activated by ``extra``.

    A requirement is included only when its marker is satisfied *because* of the
    extra: it must evaluate true with the extra set and false with no extra. This
    excludes base dependencies that merely carry an environment marker.

    Args:
        dist: Distribution name whose metadata is read.
        extra: Extra name whose dependencies are wanted.

    Returns:
        The parsed requirements activated by ``extra`` in the current environment.
    """
    target = str(canonicalize_name(extra))
    result: list[Requirement] = []
    for spec in metadata.requires(dist) or []:
        req = Requirement(spec)
        marker = req.marker
        if marker is None:
            continue
        if marker.evaluate({"extra": target}) and not marker.evaluate({"extra": ""}):
            result.append(req)
    return result
