"""Walk preset / OptimizationConfig search-space dicts and extract module info.

This module is the only place that knows the nested shape of the preset YAML:
``search_space -> list of nodes -> each node has its own search_space -> list of
module entries``. All other modules in the package consume the flattened
``(node_idx, node_type, entry)`` triples this file yields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def _extract_model_names(module_entry: dict[str, Any]) -> list[str]:
    """Pull model name(s) from a search-space module entry.

    Each module entry can declare zero or more model candidates under
    ``classification_model_config`` and/or ``embedder_config``; both keys may be
    a single dict or a list of dicts, and only entries with ``model_name`` are
    kept.
    """
    candidates: list[str] = []
    cfg = module_entry.get("classification_model_config")
    if isinstance(cfg, list):
        candidates.extend(c["model_name"] for c in cfg if isinstance(c, dict) and c.get("model_name"))
    elif isinstance(cfg, dict) and cfg.get("model_name"):
        candidates.append(cfg["model_name"])
    embedder_cfg = module_entry.get("embedder_config")
    if isinstance(embedder_cfg, list):
        candidates.extend(c["model_name"] for c in embedder_cfg if isinstance(c, dict) and c.get("model_name"))
    elif isinstance(embedder_cfg, dict) and embedder_cfg.get("model_name"):
        candidates.append(embedder_cfg["model_name"])
    return candidates


def _max_int(value: Any, default: int) -> int:  # noqa: ANN401
    """Coerce a search-space distribution descriptor into an int upper bound.

    Accepts a plain int, a list of candidate values (returns the max), or an
    Optuna-style ``{"low": ..., "high": ...}`` range dict (returns the high end).
    Anything unparseable falls back to ``default``.
    """
    if value is None:
        return default
    if isinstance(value, list) and value:
        return max(int(x) for x in value)
    if isinstance(value, dict):
        return int(value.get("high", default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _module_cardinality(entry: dict[str, Any]) -> int | None:
    """Approximate number of unique configurations the module entry can produce.

    Returns:
        * ``1`` when every tunable field is a singleton (list of length 1, or a
          plain scalar). Signal that HPO would rediscover the same config every
          trial — real optuna dedupes, so effective n_trials = 1.
        * ``N`` when the cardinality is finite and bounded (product of list
          lengths across categorical fields, capped at :data:`_CARDINALITY_CAP`
          to avoid overflow on large multi-list grids).
        * ``None`` when any field declares a continuous range (``{low, high}``
          dict) — treated as "unbounded" so the caller falls back to the full
          node-level n_trials.

    Ignores ``module_name`` (fixed, not a search dim), reserved keys like
    ``target_metric``, and any non-tunable scalar keys that are already
    single values.
    """
    _RESERVED = {"module_name", "target_metric"}
    _CARDINALITY_CAP = 10_000
    product = 1
    for key, value in entry.items():
        if key in _RESERVED:
            continue
        if isinstance(value, dict):
            # Optuna-style range descriptor with low/high => continuous; treat
            # as unbounded (many possible samples).
            if "low" in value and "high" in value:
                return None
            # Non-range dict (e.g. nested config) counts as 1 — the dict
            # itself is fixed unless it wraps a list.
            continue
        if isinstance(value, list):
            if not value:
                continue
            # A list of dicts (like classification_model_config: [{...}, {...}])
            # still counts as N candidates. Length 1 = singleton.
            product *= max(1, len(value))
            if product >= _CARDINALITY_CAP:
                return _CARDINALITY_CAP
        # Plain scalar (str/int/float/bool/None) is a singleton — contributes 1.
    return product


def _walk_modules_indexed(
    search_space: list[dict[str, Any]],
) -> Iterable[tuple[int, str, dict[str, Any]]]:
    """Yield ``(node_index, node_type, module_entry)`` triples.

    The index lets the resource phase bound per-node max cost — see
    ``dump_modules`` accounting in ``_resource.py``.
    """
    for node_idx, node in enumerate(search_space or []):
        node_type = node.get("node_type", "?")
        for entry in node.get("search_space", []) or []:
            yield node_idx, node_type, entry


def _walk_modules(search_space: list[dict[str, Any]]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield ``(node_type, module_entry)`` pairs — index-agnostic view."""
    for _, node_type, entry in _walk_modules_indexed(search_space):
        yield node_type, entry
