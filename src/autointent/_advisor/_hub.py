"""HF Hub metadata lookups + warm-cache probe.

Memoized per-process. Offline-safe: every probe falls back to a
heuristic value rather than raising. The advisor flips the report's
``low_confidence`` flag when a fallback is taken.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from huggingface_hub import HfApi, scan_cache_dir, try_to_load_from_cache

logger = logging.getLogger(__name__)

# Coarse heuristic estimates keyed on name fragments. Used only when HF Hub
# is unreachable and we can't get safetensors metadata. Values in millions.
_NAME_HEURISTICS = [
    (re.compile(r"(?i)(deberta|roberta|bert).*(xxlarge|huge)"), 1_500),
    (re.compile(r"(?i)(deberta|roberta|bert).*xlarge"), 750),
    (re.compile(r"(?i)(deberta|roberta|bert).*large"), 350),
    (re.compile(r"(?i)e5.*large"), 560),
    (re.compile(r"(?i)e5.*small"), 33),
    (re.compile(r"(?i)mpnet"), 110),
    (re.compile(r"(?i)minilm"), 33),
    (re.compile(r"(?i)distil"), 66),
    (re.compile(r"(?i)small"), 60),
    (re.compile(r"(?i)base"), 110),
    (re.compile(r"(?i)large"), 350),
]


@dataclass
class ModelMeta:
    name: str
    params_millions: float
    weight_bytes_per_param: int
    total_file_bytes: int
    cached_locally: bool
    confidence: str  # "hub" | "heuristic"

    @property
    def disk_gb(self) -> float:
        return self.total_file_bytes / (1024**3)

    @property
    def weights_gb(self) -> float:
        return (self.params_millions * 1_000_000 * self.weight_bytes_per_param) / (1024**3)


@lru_cache(maxsize=1)
def hub_reachable(timeout_s: float = 2.0) -> bool:
    """Single up-front probe. Memoized per process."""
    try:
        HfApi().list_models(limit=1)
    except Exception as e:  # noqa: BLE001
        logger.debug("HF Hub probe failed: %s", e)
        return False
    return True


def _heuristic_params_millions(model_name: str) -> float:
    for pattern, m in _NAME_HEURISTICS:
        if pattern.search(model_name):
            return float(m)
    return 110.0  # generic BERT-base default


def _is_warm_cached(model_name: str) -> bool:
    """True when the weight shard is present in the local HF cache."""
    weight_files = ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]
    for fname in weight_files:
        path = try_to_load_from_cache(model_name, fname)
        if isinstance(path, str):
            return True

    # sharded models won't match the single-file probe; fall back to a scan
    try:
        cache = scan_cache_dir()
    except Exception as e:  # noqa: BLE001
        logger.debug("scan_cache_dir failed: %s", e)
        return False
    return any(repo.repo_id == model_name for repo in cache.repos)


def _hub_metadata(model_name: str) -> ModelMeta | None:
    try:
        info = HfApi().model_info(model_name, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        logger.debug("model_info(%s) failed: %s", model_name, e)
        return None

    params_millions = 0.0
    weight_bytes_per_param = 4
    safetensors = getattr(info, "safetensors", None)
    if safetensors is not None:
        params_total = getattr(safetensors, "total", None) or sum(
            getattr(safetensors, "parameters", {}).values() or [0]
        )
        if params_total:
            params_millions = params_total / 1_000_000
            params_map: dict[str, Any] = getattr(safetensors, "parameters", {}) or {}
            if any("F16" in k or "BF16" in k for k in params_map):
                weight_bytes_per_param = 2

    total_file_bytes = 0
    for sibling in getattr(info, "siblings", []) or []:
        size = getattr(sibling, "size", None)
        if size:
            total_file_bytes += int(size)

    # Track whether either size came from the Hub or from the name-pattern fallback;
    # if any field was filled by heuristic, downgrade confidence so the report flips
    # low_confidence rather than misreporting hub-grade accuracy.
    confidence = "hub"
    if params_millions == 0:
        params_millions = _heuristic_params_millions(model_name)
        confidence = "heuristic"

    if total_file_bytes == 0:
        total_file_bytes = int(params_millions * 1_000_000 * weight_bytes_per_param)
        confidence = "heuristic"

    return ModelMeta(
        name=model_name,
        params_millions=params_millions,
        weight_bytes_per_param=weight_bytes_per_param,
        total_file_bytes=total_file_bytes,
        cached_locally=_is_warm_cached(model_name),
        confidence=confidence,
    )


def _heuristic_metadata(model_name: str) -> ModelMeta:
    params_millions = _heuristic_params_millions(model_name)
    weight_bytes_per_param = 4
    total_file_bytes = int(params_millions * 1_000_000 * weight_bytes_per_param)
    return ModelMeta(
        name=model_name,
        params_millions=params_millions,
        weight_bytes_per_param=weight_bytes_per_param,
        total_file_bytes=total_file_bytes,
        cached_locally=_is_warm_cached(model_name),
        confidence="heuristic",
    )


@lru_cache(maxsize=64)
def resolve_model(model_name: str) -> ModelMeta:
    """Resolve metadata for a single model name. Memoized per process.

    Always returns a value — never raises — so the advisor can keep going
    on offline machines or for unknown checkpoints.
    """
    if model_name.startswith("local:") or os.path.isabs(model_name):
        return ModelMeta(
            name=model_name,
            params_millions=_heuristic_params_millions(model_name),
            weight_bytes_per_param=4,
            total_file_bytes=0,
            cached_locally=True,
            confidence="heuristic",
        )

    if hub_reachable():
        meta = _hub_metadata(model_name)
        if meta is not None:
            return meta

    return _heuristic_metadata(model_name)
