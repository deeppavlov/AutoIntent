"""Validate optional-extra dependencies from installed package metadata.

The :func:`require` guard checks that every dependency of an ``autointent`` extra
is installed and version-satisfied. It reads the metadata that the build baked into
the installed distribution (via :mod:`importlib.metadata`) rather than the source
``pyproject.toml``, which is not shipped in the wheel. Nested extras are resolved
recursively, so e.g. the ``transformers`` extra (``transformers[torch]``)
transitively requires ``accelerate`` and that is checked too.
"""

from __future__ import annotations

from functools import cache
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


def _resolve(dist: str, extra: str, seen: set[tuple[str, str]]) -> list[Requirement]:
    """Recursively collect every leaf requirement activated by ``dist[extra]``.

    Each activated requirement is returned for version checking, and any nested
    extras it declares (e.g. ``transformers[torch]``) are resolved in turn.

    Args:
        dist: Distribution name to start from.
        extra: Extra name to resolve.
        seen: Visited ``(dist, extra)`` pairs, used to break dependency cycles.

    Returns:
        The flattened list of requirements to validate.
    """
    key = (str(canonicalize_name(dist)), str(canonicalize_name(extra)))
    if key in seen:
        return []
    seen.add(key)

    leaves: list[Requirement] = []
    for req in _iter_extra_reqs(dist, extra):
        leaves.append(req)
        for nested in req.extras:
            leaves.extend(_resolve(req.name, nested, seen))
    return leaves


@cache
def _resolve_cached(dist: str, extra: str) -> tuple[Requirement, ...]:
    """Memoized :func:`_resolve`; the metadata graph shape is stable per process.

    Args:
        dist: Distribution name to start from.
        extra: Extra name to resolve.

    Returns:
        The resolved requirements as an immutable tuple.
    """
    return tuple(_resolve(dist, extra, set()))


def _provides_extras(dist: str) -> set[str]:
    """Return the normalized set of extras declared by ``dist``.

    Args:
        dist: Distribution name whose metadata is read.

    Returns:
        Normalized extra names from the distribution's ``Provides-Extra`` metadata.
    """
    md = metadata.metadata(dist)
    return {str(canonicalize_name(e)) for e in (md.get_all("Provides-Extra") or [])}


def require(extra: str, *, dist: str = _DIST) -> None:
    """Ensure every dependency of an ``autointent`` extra is installed and current.

    Args:
        extra: The extra to validate, e.g. ``"transformers"``.
        dist: Distribution that declares the extra. Defaults to ``"autointent"``.

    Raises:
        ValueError: If ``dist`` declares no such ``extra`` (typically a typo).
        ImportError: If any required dependency is missing or its installed version
            does not satisfy the constraint declared in the metadata.
    """
    known = _provides_extras(dist)
    if str(canonicalize_name(extra)) not in known:
        msg = f"'{dist}' declares no extra '{extra}'. Known extras: {', '.join(sorted(known))}."
        raise ValueError(msg)

    problems: list[str] = []
    for req in _resolve_cached(dist, extra):
        problem = _check(req)
        if problem is not None and problem not in problems:
            problems.append(problem)

    if problems:
        bullets = "\n".join(f"  - {p}" for p in problems)
        msg = (
            f"Feature requires extra '{extra}', but dependencies are missing or outdated:\n"
            f"{bullets}\n"
            f"Install with: pip install '{dist}[{extra}]'"
        )
        raise ImportError(msg)
