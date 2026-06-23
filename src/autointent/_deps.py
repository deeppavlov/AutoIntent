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
from typing import Literal

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_DIST = "autointent"

# Names of the optional-dependency extras autointent declares, mirrored from the
# installed ``Provides-Extra`` metadata (and thus pyproject's
# [project.optional-dependencies]). Typing ``require``'s parameter with this makes
# mypy reject misspelled extra names at call sites; the runtime check in ``require``
# stays the source of truth (mypy isn't run at runtime, and ``dist`` overrides or
# dynamic calls bypass static typing). Kept in sync with the real metadata by
# tests/test_deps.py::test_extra_literal_matches_real_metadata.
Extra = Literal[
    "catboost",
    "codecarbon",
    "dspy",
    "fastapi",
    "fastmcp",
    "openai",
    "opensearch",
    "peft",
    "sentence-transformers",
    "transformers",
    "vllm",
    "wandb",
]


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

    Args:
        dist: Distribution name whose metadata is read.
        extra: Extra name whose dependencies are wanted.

    Returns:
        The parsed requirements activated by ``extra`` in the current environment,
        or an empty list if ``dist`` is not installed (its metadata is unavailable).
    """
    target = str(canonicalize_name(extra))
    result: list[Requirement] = []
    try:
        reqs = metadata.requires(dist)
    except metadata.PackageNotFoundError:
        # `dist` itself isn't installed, so we can't read its nested-extra
        # requirements. That's fine: the parent requirement that led us to recurse
        # here (e.g. `transformers[torch]`) was already collected by the caller and
        # `_check` will flag it as "not installed", producing the proper aggregated
        # ImportError with the install hint -- rather than letting a raw
        # PackageNotFoundError leak out of the resolver.
        return []
    for spec in reqs or []:
        req = Requirement(spec)
        # `req.marker` is the parsed `;` clause of the PEP 508 requirement (a
        # packaging Marker), or None when the requirement has no `;` clause. There
        # are three cases:
        # (1) no marker -> an unconditional base dependency;
        # (2) a marker that references `extra` -> belongs to an extra;
        # (3) a marker with only environment conditions (e.g. `python_version < "3.9"`)
        # -> still a base dependency, just platform-conditional.
        # So "has a marker" does NOT mean "belongs to an extra";
        marker = req.marker
        # Here we cancel out case (1)
        if marker is None:
            continue
        # `marker.evaluate(env)` resolves the whole boolean expression to a bool,
        # filling any keys we omit (python_version, sys_platform, ...) from the
        # running interpreter. A single `evaluate({"extra": target})` is not enough
        # to prove membership: an env-conditional base dep also passes it, because
        # its truth comes from the environment and the `extra` key is ignored.
        # The discriminator is the second evaluation: a *true* extra dependency
        # flips active -> inactive when the extra is removed, whereas a base dep is
        # unaffected. So "active with the extra AND inactive with no extra" means
        # "active *because of* this extra", which keeps extra members and drops
        # base deps. We always pass `extra` explicitly (`""` = base install, no
        # extras) since a marker that references `extra` can't be evaluated without it.
        # So here we cancel out case (3)
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


def require(extra: Extra, *, dist: str = _DIST) -> None:
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
