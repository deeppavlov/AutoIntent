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

from autointent.advisor._report import Severity

if TYPE_CHECKING:
    from autointent.advisor._hub import ModelMeta
    from autointent.advisor._report import DatasetStats


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
    """Weight-side VRAM: weights + grads + optimizer state.

    Pessimistic upper bound by mode: 1.3x inference, 1.3x + 0.5 GB lora
    adapters, 4.5x full finetune (textbook 4W + fragmentation/workspaces
    slack).
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
    """Activation memory per sample.

    Training: 34 B/token/layer (Korthikanti 2022 upper bound, standard
    attention). Inference: 8 B/token (only 1-2 layers' outputs in flight).
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
    """Total VRAM: weights + grads + optimizer state + activations x batch.

    Safety margin: 1.20 for training (backward transients, eval sweep,
    allocator fragmentation), 1.10 for inference (no backward).
    """
    base = _weights_vram_for_transformer(meta, mode)
    if batch_size <= 0:
        return base
    is_training = mode != "inference"
    per_sample = _activations_gb_per_sample(meta, seq_len, is_training=is_training)
    safety = 1.20 if is_training else 1.10
    return (base + per_sample * batch_size) * safety


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


# Sustained TFLOPS per device class — real HF-Trainer MFU (~20% on A100),
# not peak spec sheet. Advisor upper-bounds, so pessimistic values here.
_DEVICE_TFLOPS = {
    "high-gpu": 60.0,   # A100 / H100
    "mid-gpu": 20.0,    # V100 / RTX 3090 / A6000
    "low-gpu": 7.0,     # T4 / RTX 3060 / 8 GB consumer card
    "apple-silicon": 4.0,  # M1/M2/M3 GPU cores
    "cpu": 0.05,        # single-thread modern x86 with MKL
}
_DEFAULT_TFLOPS = 7.0  # unknown device → treat as low-GPU

# HF Trainer overhead: eval sweeps, save syncs, dataloader idle. The raw
# FLOPs formula only counts optimizer steps.
_TRAINER_OVERHEAD_MULT = 1.35


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
    """Transformer training wall-time in hours.

    Per-step FLOPs = 6 x params x batch x seq_len (fwd+bwd), ÷ sustained
    device TFLOPS, x total steps x trainer overhead.
    """
    steps_per_epoch = max(1, n_samples // max(1, batch_size))
    total_steps = n_trials * epochs * steps_per_epoch
    # 6x factor: ~2x for fwd matmul + ~4x for bwd (grad wrt input + grad wrt weight).
    step_flops = 6.0 * params_millions * 1e6 * batch_size * seq_len
    tflops = _DEVICE_TFLOPS.get(device_class, _DEFAULT_TFLOPS)
    step_seconds = step_flops / (tflops * 1e12)
    return (total_steps * step_seconds * _TRAINER_OVERHEAD_MULT) / 3600.0


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


def _ram_for_module(meta: ModelMeta, stats: DatasetStats, *, mode: str = "inference") -> float:
    """RAM upper bound: weights x mode_mult + tokenized text (n_samples x avg_tokens x 4 B).

    Mode multiplier: 1.3 inference, 1.5 lora, 4.5 full-finetune (Adam
    mirrors weights on host too).
    """
    if mode == "inference":
        weights_mult = 1.3
    elif mode == "lora":
        weights_mult = 1.5
    else:
        weights_mult = 4.5
    tokens_gb = (stats.n_samples * stats.avg_tokens * 4) / _BYTES_PER_GB
    return meta.weights_gb * weights_mult + tokens_gb


def _embedding_cache_disk_gb(n_samples: int, hidden_size: int) -> float:
    """Disk footprint of one fp32 cached embedding file: ``n_samples x hidden_size x 4``."""
    return (n_samples * hidden_size * 4) / _BYTES_PER_GB


# Wall-time coefficients calibrated on 1-thread CPU (OMP_NUM_THREADS=1),
# seconds per fit-work-unit. Typical L-BFGS iteration count baked in.
_LINEAR_CPU_S_PER_SAMPLE_FEATURE = 1.2e-9
_CATBOOST_CPU_S_PER_SAMPLE_FEATURE_ITER = 1e-9
_CATBOOST_GPU_SPEEDUP = 10.0
_LOGREG_CV_MULTIPLIER = 31  # sklearn default: Cs=10 x cv=3 + 1 final refit
_CATBOOST_DEFAULT_BINS = 254  # CatBoost `border_count` default
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
    max_iter: int,  # noqa: ARG001 — API stability; typical L-BFGS convergence baked into coeff
    cv_multiplier: int,
    class_multiplier: int,
) -> float:
    """LogisticRegression wall time.

    O(n_samples x features x classes x cv) per fit; typical L-BFGS
    convergence absorbed into the calibration constant.
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


def _ram_for_sklearn(
    *,
    stats: DatasetStats,
    embedder_dim: int,
    n_estimators: int,
    max_depth: int,
    n_jobs: int,
) -> float:
    """RandomForest RAM: (feature matrix + trees) x n_jobs.

    joblib workers each hold a full copy.
    """
    per_worker_data = stats.n_samples * embedder_dim * 8  # fp64 default
    n_leaves = min(2**max_depth, stats.n_samples) if max_depth > 0 else stats.n_samples
    per_worker_trees = n_estimators * n_leaves * _CATBOOST_BYTES_PER_TREE_NODE
    return float(((per_worker_data + per_worker_trees) * max(1, n_jobs)) / _BYTES_PER_GB)


def _embedder_load_ram_gb(meta: ModelMeta | None) -> float:
    """Aggregate-level RAM penalty when a classic preset uses an embedder.

    Added on top of the max-driver RAM because embedder + multiple classic
    scorers coexist in memory. Uses fp32 weights (transformers up-casts at
    load) x 3.5 for weights + activation buffers + framework slack.
    """
    if meta is None:
        return 0.0
    return ((meta.total_params * 4) / _BYTES_PER_GB) * 3.5


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


# === CNN / RNN scorers ===================================================
# Small torch models (TextCNN, LSTM) trained from scratch on token ids.
_NN_MAX_VOCAB = 30_000  # upper bound on vocabulary size
_NN_DEFAULT_SEQ_LEN = 50  # VocabConfig.max_seq_length default
_NN_BYTES_PER_PARAM = 4  # fp32
_NN_TRAIN_ACT_BYTES_PER_UNIT = 16  # ~4x backward + optimizer overhead


def _cnn_param_count(*, embed_dim: int, num_filters: int, n_kernels: int, n_classes: int) -> int:
    """TextCNN params: embedding + conv (kernel width ~5) + fc."""
    return (
        _NN_MAX_VOCAB * embed_dim
        + num_filters * embed_dim * n_kernels * 5
        + num_filters * n_kernels * max(1, n_classes)
    )


def _rnn_param_count(*, embed_dim: int, hidden_dim: int, n_classes: int) -> int:
    """LSTM classifier params: embedding + 4-gate LSTM cell + fc."""
    return (
        _NN_MAX_VOCAB * embed_dim
        + 4 * hidden_dim * (embed_dim + hidden_dim + 1)
        + hidden_dim * max(1, n_classes)
    )


def _vram_for_nn(*, params: int, batch_size: int, hidden_dim: int) -> float:
    """Weights + 3x optimizer/grads + activations.

    Same fp32 upper bound as transformers, smaller hidden dim (embed_dim /
    num_filters).
    """
    weights_gb = (params * _NN_BYTES_PER_PARAM) / _BYTES_PER_GB
    activations_gb = (
        batch_size * _NN_DEFAULT_SEQ_LEN * hidden_dim * _NN_TRAIN_ACT_BYTES_PER_UNIT
    ) / _BYTES_PER_GB
    return 4 * weights_gb + activations_gb


def _ram_for_nn(*, params: int, stats: DatasetStats) -> float:
    """Weights + tokenized text (int32 ids)."""
    return ((params * _NN_BYTES_PER_PARAM) + (stats.n_samples * _NN_DEFAULT_SEQ_LEN * 4)) / _BYTES_PER_GB


def _time_for_nn(
    *,
    n_trials: int,
    epochs: int,
    batch_size: int,
    n_samples: int,
    params_millions: float,
    device_class: str,
) -> float:
    """Reuse transformer FLOPs formula.

    Small models slightly under-predict since they're memory-bandwidth-bound,
    but within 2x for cost ranking.
    """
    return _time_for_transformer(
        n_trials=n_trials,
        epochs=epochs,
        batch_size=batch_size,
        seq_len=_NN_DEFAULT_SEQ_LEN,
        n_samples=n_samples,
        params_millions=params_millions,
        device_class=device_class,
    )


def _floor_to_power_of_two(n: int) -> int:
    """Largest power of two <= ``n``; returns 0 when ``n < 1``."""
    if n < 1:
        return 0
    power = 1
    while power * 2 <= n:
        power *= 2
    return power
