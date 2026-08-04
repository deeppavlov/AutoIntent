"""Pure cost-estimate formulas — VRAM, RAM, time, severity, model shape.

No I/O, no logging, no orchestration. Each formula docstring links to the
reference it was calibrated against so a reviewer can follow each coefficient
back to its source.

Conventions:
  * All ``*_gb`` results use the binary GiB convention (1024**3 bytes per GB) —
    matches the rest of the advisor's byte->GB conversions.
  * All ``*_hours`` results assume the GPU baseline of ~1 second per step;
    CPU runs pay a flat slowdown factor (see ``_time_for_transformer``).
  * "fp32 worst case" — we deliberately ignore lower-precision / FlashAttention /
    quantization optimizations, per the advisor's "pessimistic upper bound" contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from autointent._advisor._report import Severity

if TYPE_CHECKING:
    from autointent._advisor._hub import ModelMeta
    from autointent._advisor._report import DatasetStats


_BYTES_PER_GB = 1024**3
_DEFAULT_SEQ_LEN = 128

# Fallback architecture shape (BERT-base) used only when the model's actual
# config.json couldn't be fetched from HF Hub — see _hub._shape_from_config.
_DEFAULT_HIDDEN = 768
_DEFAULT_LAYERS = 12

_TIGHT_RATIO = 0.9
_MULTICLASS_THRESHOLD = 2


def _classify_severity(estimate: float, budget: float) -> Severity:
    """Map a ``(estimate, budget)`` pair onto a Severity bucket.

    * AMPLE: ``estimate <= 0`` OR ``ratio < _TIGHT_RATIO``
    * TIGHT: ``budget <= 0`` OR ``_TIGHT_RATIO <= ratio < 1``
    * OVER:  ``ratio >= 1``
    """
    if estimate <= 0:
        return Severity.AMPLE
    if budget <= 0:
        return Severity.TIGHT
    ratio = estimate / budget
    if ratio >= 1:
        return Severity.OVER
    if ratio >= _TIGHT_RATIO:
        return Severity.TIGHT
    return Severity.AMPLE


def _weights_vram_for_transformer(meta: ModelMeta, mode: str) -> float:
    """Weight-side VRAM in GB — weights + grads + optimizer state. Excludes activations.

    Returns a deliberately pessimistic upper bound, matching the advisor's
    "heuristic upper bound, not measurement" contract.

    Modes:
      * ``inference``: forward only — weights + ~30% intermediate-tensor overhead.
      * ``lora``: frozen base + small trainable adapters + their grads/optimizer (~0.5 GB).
      * ``full-finetune`` (default): the textbook 4W (weights + grads + Adam m + Adam v).
        We use 4.5W to leave headroom for loss-scale buffers, allocator fragmentation,
        cuDNN workspaces, and gradient-accumulation buffers — none of which the textbook
        4W accounting captures.
    """
    weights_gb = meta.weights_gb
    if mode == "inference":
        return weights_gb * 1.3
    if mode == "lora":
        return weights_gb * 1.3 + 0.5
    return weights_gb * 4.5


def _activations_gb_per_sample(
    meta: ModelMeta | None,
    seq_len: int,
    *,
    is_training: bool,
) -> float:
    """Heuristic activation memory per sample.

    Training uses **34 bytes/token/layer** as a pessimistic upper bound —
    Korthikanti et al. (2022, "Reducing Activation Recomputation ...") derive
    this for standard attention: the linear-layer activations account for ~11B
    and the attention matrix + intermediate tensors add ~23B. FlashAttention
    kernels drop the attention-matrix term (~12 B/token/layer total), but we
    can't detect at preflight time whether the user's stack will use them, so
    the upper bound is the safe choice.

    Inference: only 1-2 layers' outputs are kept in flight at once. 8 B/token
    covers fp32 hidden (4B) plus a bit of intermediate slack.

    An earlier revision used 16 B/token/layer for training; that under-predicted
    real deberta-v3-large VRAM by ~2x at bs=128 (measured 13.1 GB, predicted
    ~11.5 GB), which is unsafe for an OOM-avoidance tool. See ``interpretation.md``
    (2026-07-19) for the calibration data.
    """
    hidden = _embedder_dim(meta)
    training_bytes_per_token_per_layer = 34
    inference_bytes_per_token = 8
    if is_training:
        bytes_per_sample = seq_len * hidden * _n_layers(meta) * training_bytes_per_token_per_layer
    else:
        bytes_per_sample = seq_len * hidden * inference_bytes_per_token
    return bytes_per_sample / _BYTES_PER_GB


def _vram_for_transformer(
    meta: ModelMeta,
    mode: str,
    *,
    batch_size: int = 0,
    seq_len: int = _DEFAULT_SEQ_LEN,
) -> float:
    """Total VRAM in GB: weights + grads + optimizer state + activations x batch.

    Activation accounting differs by mode — training keeps per-layer outputs for
    backward; inference only needs one or two layers in flight.
    """
    base = _weights_vram_for_transformer(meta, mode)
    if batch_size <= 0:
        return base
    per_sample = _activations_gb_per_sample(meta, seq_len, is_training=mode != "inference")
    return base + per_sample * batch_size


def _max_fitting_batch_size(
    *,
    weight_vram_gb: float,
    vram_budget_gb: float,
    per_sample_gb: float,
) -> int:
    """Largest batch that keeps total VRAM under the AMPLE/TIGHT threshold.

    Returns 0 when even the weights blow the budget. Result is rounded down to
    the nearest power of two
    """
    if per_sample_gb <= 0:
        return 0
    target_vram = vram_budget_gb * _TIGHT_RATIO
    available_for_activations = target_vram - weight_vram_gb
    if available_for_activations <= 0:
        return 0
    return _floor_to_power_of_two(int(available_for_activations / per_sample_gb))


# Sustained TFLOPS per device class — real MFU (model-FLOPs utilization) at
# training batch sizes, NOT peak spec sheet numbers. Numbers reflect ~30-50%
# MFU which is typical for BERT-scale training with FA2 / cuDNN kernels.
# Source: MLPerf training results + community benchmarks (2024-2025).
_DEVICE_TFLOPS = {
    "high-gpu": 150.0,  # A100 / H100 fp16
    "mid-gpu": 45.0,    # V100 / RTX 3090 / A6000
    "low-gpu": 15.0,    # T4 / RTX 3060 / 8 GB consumer card
    "apple-silicon": 8.0,  # M1/M2/M3 GPU cores
    "cpu": 0.1,         # single-thread modern x86 with MKL
}
_DEFAULT_TFLOPS = 15.0  # unknown device → treat as low-GPU


def _time_for_transformer(
    *,
    n_trials: int,
    epochs: int,
    batch_size: int,
    seq_len: int,
    n_samples: int,
    params_millions: float,
    device_class: str,
) -> float:
    """Transformer training time in hours, from per-step FLOPs / device TFLOPS.

    Per-step FLOPs ≈ 6 x params x batch_size x seq_len (2 for forward mul-add,
    3-4x for backward). Divided by sustained device TFLOPS to get wall-time per
    step, then multiplied by (steps x epochs x n_trials).

    Replaces an earlier "1 second per step" heuristic, which was ~10x too high
    on A100 and identical for MPS vs CUDA (predicted times were the same on
    both while real times differed ~7x — see interpretation.md 2026-07-19).
    """
    steps_per_epoch = max(1, n_samples // max(1, batch_size))
    total_steps = n_trials * epochs * steps_per_epoch
    # 6x factor: ~2x for fwd matmul + ~4x for bwd (grad wrt input + grad wrt weight).
    step_flops = 6.0 * params_millions * 1e6 * batch_size * seq_len
    tflops = _DEVICE_TFLOPS.get(device_class, _DEFAULT_TFLOPS)
    step_seconds = step_flops / (tflops * 1e12)
    return (total_steps * step_seconds) / 3600.0


def _n_layers(meta: ModelMeta | None) -> int:
    """Layer count from the model's ``config.json``; falls back to BERT-base when absent."""
    if meta is not None and meta.n_layers is not None:
        return meta.n_layers
    return _DEFAULT_LAYERS


def _embedder_dim(meta: ModelMeta | None) -> int:
    """Hidden size from the model's ``config.json``; falls back to BERT-base when absent."""
    if meta is not None and meta.hidden_size is not None:
        return meta.hidden_size
    return _DEFAULT_HIDDEN


def _largest_embedder(seen_models: dict[str, ModelMeta]) -> ModelMeta | None:
    """Return the largest model in ``seen_models`` by parameter count, or None if empty."""
    if not seen_models:
        return None
    return max(seen_models.values(), key=lambda m: m.total_params)


def _ram_for_module(meta: ModelMeta, stats: DatasetStats) -> float:
    """RAM in GB. Loose upper bound: weights + tokenized text in memory.

    Tokenized text is approximated as ``n_samples x avg_tokens x 4 bytes``
    (BPE/WordPiece token ids fit in int32). The 4 bytes/token bound is tight
    enough for the report's purposes and intentionally ignores any preprocessing
    artefacts (attention masks, position ids, etc.) since they're bounded by the
    same factor.
    """
    return meta.weights_gb + (stats.n_samples * stats.avg_tokens * 4) / _BYTES_PER_GB


def _embedding_cache_disk_gb(n_samples: int, hidden_size: int) -> float:
    """Disk footprint of one fp32 cached embedding file: ``n_samples x hidden_size x 4``."""
    return (n_samples * hidden_size * 4) / _BYTES_PER_GB


# Wall-time coefficients calibrated against measured fits on 1-thread CPU
# (OMP_NUM_THREADS=1). Values represent seconds per per-fit-work-unit and
# already absorb the number of L-BFGS iterations the optimizer typically
# takes to converge (~50) — so the ``max_iter`` upper bound does NOT enter
# the formula directly. Historical formula also scaled by ``max_iter`` which
# multi-cent-ordered-over-predicted (137 h vs measured ~30 s = ~15000x on
# banking77 × 1024-dim e5-large × 77 classes × cv=3).
#
# Calibration point (reviewer's res-adapt-ckeck/a100 run, warm cache):
#   classic-light linear on banking77 (n=10003, dim=1024, cls=77, cv_mult=31)
#   measured ~30 s per fit x n_trials=20 = ~10 min total = ~0.17 h.
#   Formula: 20 x 1.2e-9 x 10003 x 1024 x 31 x 77 = ~588 s = ~0.16 h  ✓
_LINEAR_CPU_S_PER_SAMPLE_FEATURE = 1.2e-9
_CATBOOST_CPU_S_PER_SAMPLE_FEATURE_ITER = 1e-9  # catboost is measured per iteration
_CATBOOST_GPU_SPEEDUP = 10.0
# LogisticRegressionCV defaults: Cs=10, cv=3 -> 10x3 inner fits + 1 final refit = 31.
_LOGREG_CV_MULTIPLIER = 31
# Default value of `border_count` in CatBoost (number of histogram buckets per feature).
_CATBOOST_DEFAULT_BINS = 254
# Bytes per histogram bucket / tree node — order-of-magnitude constant.
_CATBOOST_BYTES_PER_TREE_NODE = 32


def _ram_for_linear(*, stats: DatasetStats, embedder_dim: int) -> float:
    """Float64 design matrix dominates; coefficients and L-BFGS history are small."""
    data_bytes = 8.0 * stats.n_samples * embedder_dim
    coef_bytes = 8.0 * max(1, stats.n_classes) * embedder_dim
    lbfgs_bytes = 10.0 * 8.0 * embedder_dim
    return (data_bytes + coef_bytes + lbfgs_bytes) / _BYTES_PER_GB


def _time_for_linear(
    *,
    n_trials: int,
    n_samples: int,
    embedder_dim: int,
    max_iter: int,  # noqa: ARG001 — kept in signature for API stability; typical L-BFGS convergence is absorbed into the coefficient
    cv_multiplier: int,
    class_multiplier: int,
) -> float:
    """LogisticRegression wall time, in hours.

    Cost is ``O(n_samples x n_features x n_classes)`` per fit (sklearn's L-BFGS
    solver, iterations absorbed into the calibration constant), multiplied by the
    CV inner-fit count (31 for the default LogisticRegressionCV).

    ``max_iter`` is a per-fit upper bound, not the typical work — L-BFGS on a
    well-conditioned classifier converges long before it. Older versions of
    this formula scaled by ``max_iter`` and predicted ~1000x higher than
    reality; the constant now bakes in a typical convergence-iteration count.
    """
    seconds = (
        n_trials
        * _LINEAR_CPU_S_PER_SAMPLE_FEATURE
        * n_samples
        * embedder_dim
        * cv_multiplier
        * class_multiplier
    )
    return seconds / 3600.0


def _ram_for_catboost(*, stats: DatasetStats, n_features: int, iterations: int, depth: int) -> float:
    """CatBoost RAM = quantized data matrix + histograms + tree storage."""
    data_bytes = 4.0 * stats.n_samples * n_features
    histograms_bytes = 4.0 * n_features * _CATBOOST_DEFAULT_BINS
    trees_bytes = iterations * (2**depth) * _CATBOOST_BYTES_PER_TREE_NODE
    return float((data_bytes + histograms_bytes + trees_bytes) / _BYTES_PER_GB)


def _time_for_catboost(
    *,
    n_trials: int,
    n_samples: int,
    n_features: int,
    iterations: int,
    depth: int,
    class_multiplier: int,
    on_gpu: bool,
) -> float:
    """CatBoost wall time, in hours.

    Cost is ``O(iterations x n_samples x n_features x depth x n_classes)`` per
    fit. GPU training is ~10x faster than CPU for typical workloads per
    CatBoost's published benchmarks.
    https://catboost.ai/en/docs/concepts/speed-up-training
    """
    coeff = _CATBOOST_CPU_S_PER_SAMPLE_FEATURE_ITER
    if on_gpu:
        coeff /= _CATBOOST_GPU_SPEEDUP
    seconds = n_trials * iterations * coeff * n_samples * n_features * depth * class_multiplier
    return seconds / 3600.0


def _floor_to_power_of_two(n: int) -> int:
    """Largest power of two <= ``n``; returns 0 when ``n < 1``."""
    if n < 1:
        return 0
    power = 1
    while power * 2 <= n:
        power *= 2
    return power
