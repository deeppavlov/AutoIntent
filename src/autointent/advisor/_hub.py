"""HF Hub metadata lookups + warm-cache probe.

Memoized per-process. Offline-safe: every probe falls back to a
heuristic value rather than raising. The advisor flips the report's
``low_confidence`` flag when a fallback is taken.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir, try_to_load_from_cache

Confidence = Literal["hub", "heuristic"]

logger = logging.getLogger(__name__)

# Conservative "large-model" shape used when Hub metadata is unavailable —
# roughly deberta-v3-large / bert-large sized. Previously we defaulted to a
# BERT-base shape (110M / 768 / 12), which *under*-predicted a real deberta-large
# fit by ~2×. Because the advisor's contract is a pessimistic upper bound, the
# offline fallback needs to over-estimate small models rather than under-estimate
# large ones. Callers can still see the fallback happened via ``confidence ==
# "heuristic"`` and ``PreflightReport.low_confidence``.
_DEFAULT_HEURISTIC_PARAMS = 350_000_000
_DEFAULT_BYTES_PER_PARAM = 4
_DEFAULT_HEURISTIC_HIDDEN = 1024
_DEFAULT_HEURISTIC_LAYERS = 24
_BYTES_PER_GB = 1024**3  # using the binary GiB convention everywhere in the advisor


@dataclass
class ModelMeta:
    name: str
    total_params: int
    weight_bytes_per_param: float
    total_file_bytes: int
    cached_locally: bool
    confidence: Confidence
    hidden_size: int | None = None
    n_layers: int | None = None

    @property
    def disk_gb(self) -> float:
        return self.total_file_bytes / _BYTES_PER_GB

    @property
    def weights_gb(self) -> float:
        return (self.total_params * self.weight_bytes_per_param) / _BYTES_PER_GB


def _shape_from_config(model_name: str) -> tuple[int | None, int | None]:
    """Return ``(hidden_size, num_hidden_layers)`` straight from the model's config.json.

    ``hf_hub_download`` caches the file after the first call, so repeated lookups
    in the same process (or across CLI invocations) hit local disk. Returns
    ``(None, None)`` on any failure — the advisor stays best-effort.
    """
    try:
        path = hf_hub_download(model_name, "config.json")
    except Exception as e:  # noqa: BLE001
        logger.debug("config.json download(%s) failed: %s", model_name, e)
        return None, None
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("config.json parse(%s) failed: %s", model_name, e)
        return None, None
    # Cover the common HF naming variants: BERT/Llama/Gemma use hidden_size +
    # num_hidden_layers; T5/MT5 use d_model + num_layers; GPT-2/Neo use n_embd + n_layer.
    hidden = cfg.get("hidden_size") or cfg.get("d_model") or cfg.get("n_embd")
    layers = cfg.get("num_hidden_layers") or cfg.get("num_layers") or cfg.get("n_layer")
    return int(hidden) if hidden else None, int(layers) if layers else None


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
    # Bytes-per-element for safetensors dtype strings. Used to convert the per-dtype
    # parameter counts (info.safetensors.parameters) into a weighted average
    # bytes-per-param when a checkpoint stores tensors in multiple dtypes.
    _dtype_bytes: dict[str, int] = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }

    total_params = 0
    weight_bytes_per_param: float = _DEFAULT_BYTES_PER_PARAM
    if info.safetensors is not None:
        params_by_dtype = info.safetensors.parameters or {}
        total_params = info.safetensors.total or sum(params_by_dtype.values())
        if total_params:
            total_weight_bytes = sum(
                _dtype_bytes.get(dtype, _DEFAULT_BYTES_PER_PARAM) * count for dtype, count in params_by_dtype.items()
            )
            if total_weight_bytes:
                weight_bytes_per_param = total_weight_bytes / total_params

    total_file_bytes = sum(s.size for s in (info.siblings or []) if s.size)

    # Track whether either size came from the Hub or from the name-pattern fallback;
    # if any field was filled by heuristic, downgrade confidence so the report flips
    # low_confidence rather than misreporting hub-grade accuracy.
    confidence: Confidence = "hub"
    if total_params == 0:
        total_params = _DEFAULT_HEURISTIC_PARAMS
        confidence = "heuristic"

    if total_file_bytes == 0:
        total_file_bytes = int(total_params * weight_bytes_per_param)
        confidence = "heuristic"

    hidden_size, n_layers = _shape_from_config(model_name)
    if hidden_size is None or n_layers is None:
        logger.warning(
            "Could not read hidden_size / num_hidden_layers from config.json for %s; "
            "activation-memory estimates will fall back to CONSERVATIVE large-model "
            "defaults (hidden=%d, layers=%d) to avoid under-predicting.",
            model_name,
            _DEFAULT_HEURISTIC_HIDDEN,
            _DEFAULT_HEURISTIC_LAYERS,
        )
        hidden_size = hidden_size or _DEFAULT_HEURISTIC_HIDDEN
        n_layers = n_layers or _DEFAULT_HEURISTIC_LAYERS
        confidence = "heuristic"

    return ModelMeta(
        name=model_name,
        total_params=total_params,
        weight_bytes_per_param=weight_bytes_per_param,
        total_file_bytes=total_file_bytes,
        cached_locally=_is_warm_cached(model_name),
        confidence=confidence,
        hidden_size=hidden_size,
        n_layers=n_layers,
    )


def _heuristic_metadata(model_name: str) -> ModelMeta:
    logger.warning(
        "Falling back to name-pattern heuristic for %s; "
        "using CONSERVATIVE large-model defaults (params=%dM, hidden=%d, layers=%d) "
        "so cost estimates upper-bound rather than under-predict.",
        model_name,
        _DEFAULT_HEURISTIC_PARAMS // 1_000_000,
        _DEFAULT_HEURISTIC_HIDDEN,
        _DEFAULT_HEURISTIC_LAYERS,
    )
    total_file_bytes = _DEFAULT_HEURISTIC_PARAMS * _DEFAULT_BYTES_PER_PARAM
    return ModelMeta(
        name=model_name,
        total_params=_DEFAULT_HEURISTIC_PARAMS,
        weight_bytes_per_param=_DEFAULT_BYTES_PER_PARAM,
        total_file_bytes=total_file_bytes,
        cached_locally=_is_warm_cached(model_name),
        confidence="heuristic",
        hidden_size=_DEFAULT_HEURISTIC_HIDDEN,
        n_layers=_DEFAULT_HEURISTIC_LAYERS,
    )


def _looks_like_local_path(model_name: str) -> bool:
    """True when ``model_name`` is a filesystem path rather than an HF Hub repo id.

    Hub repo ids match ``org/repo``; anything that starts with a path separator,
    ``~``, a relative-path prefix, or a Windows drive letter, or contains a
    backslash, is treated as a local path. We can't rely on ``Path.is_absolute()``
    alone because POSIX-style absolute paths (``/tmp/...``) are *not* absolute
    on Windows.
    """
    if model_name.startswith(("local:", "/", "~", "./", "../", "\\\\")):
        return True
    if "\\" in model_name:
        return True
    return len(model_name) >= 2 and model_name[1] == ":" and model_name[0].isalpha()  # noqa: PLR2004


@lru_cache(maxsize=64)
def resolve_model(model_name: str) -> ModelMeta:
    """Resolve metadata for a single model name. Memoized per process.

    Always returns a value — never raises — so the advisor can keep going
    on offline machines or for unknown checkpoints.
    """
    if _looks_like_local_path(model_name):
        return ModelMeta(
            name=model_name,
            total_params=_DEFAULT_HEURISTIC_PARAMS,
            weight_bytes_per_param=_DEFAULT_BYTES_PER_PARAM,
            total_file_bytes=0,
            cached_locally=True,
            confidence="heuristic",
        )

    # _hub_metadata returns None on any failure (network outage, missing repo,
    # SDK exception) so we don't need a separate up-front probe.
    meta = _hub_metadata(model_name)
    if meta is not None:
        return meta

    return _heuristic_metadata(model_name)
