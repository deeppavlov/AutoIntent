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

    Final ``* 1.20`` is a safety margin covering allocator fragmentation,
    peak transient tensors during backward, and HF-Trainer's eval-loop
    double-forward that the textbook accounting above misses. Advisor should
    upper-bound: earlier 15% margin left transformers-light on banking77 at
    8.12 GB predicted vs 8.75 GB measured (1.08x UNDER — unsafe for an OOM
    tool). Bumped to 20% + a bigger fixed CUDA baseline (see
    ``_CUDA_BASELINE_VRAM_GB``) to close the gap on large-batch training runs
    without over-inflating small ones.
    """
    base = _weights_vram_for_transformer(meta, mode)
    if batch_size <= 0:
        return base
    per_sample = _activations_gb_per_sample(meta, seq_len, is_training=mode != "inference")
    return (base + per_sample * batch_size) * 1.20


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
# training batch sizes, NOT peak spec sheet numbers. Numbers reflect ~20-30%
# MFU which is what HF Trainer actually achieves on BERT-scale workloads once
# you factor in dataloader idle, tokenization warmup, per-epoch eval, and
# checkpoint saves — all of which the raw FLOPs formula ignores. Earlier
# values (150 / 45 / 15 for high/mid/low) were closer to peak spec numbers
# and under-predicted transformers-heavy on banking77 by 1.7x (measured
# 3.17 h vs predicted 1.88 h, calibration_runs2 2026-08-07); an advisor
# should upper-bound, so pick sustained numbers that err on the side of
# over-predicting. Source: MLPerf training results + measured banking77
# runs where mean_step_s / p95_step_s → 154ms / 246ms for bert-base bs=64.
_DEVICE_TFLOPS = {
    "high-gpu": 60.0,   # A100 / H100 — sustained ~19% MFU under HF Trainer
    "mid-gpu": 20.0,    # V100 / RTX 3090 / A6000
    "low-gpu": 7.0,     # T4 / RTX 3060 / 8 GB consumer card
    "apple-silicon": 4.0,  # M1/M2/M3 GPU cores
    "cpu": 0.05,        # single-thread modern x86 with MKL
}
_DEFAULT_TFLOPS = 7.0  # unknown device → treat as low-GPU

# HF Trainer overhead — the FLOPs formula only counts optimizer steps; real
# wall-time also includes per-epoch eval sweeps, save-checkpoint syncs,
# tokenizer warmup, dataloader queue idle, and gradient-accumulation gaps.
# Factor calibrated so transformers-heavy predicts ~1.2-1.5x the measured
# 3.17 h (upper-bound stance).
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
    """Transformer training time in hours, from per-step FLOPs / device TFLOPS.

    Per-step FLOPs ≈ 6 x params x batch_size x seq_len (2 for forward mul-add,
    3-4x for backward). Divided by *sustained* device TFLOPS (not peak spec)
    to get wall-time per step, then multiplied by (steps x epochs x n_trials)
    x ``_TRAINER_OVERHEAD_MULT`` for HF-Trainer wall-clock overhead.

    Advisor contract: err on the side of over-prediction. Under-predicting
    time makes users blow through wall-clock budgets; over-predicting only
    biases them toward smaller / cheaper presets. See ``_DEVICE_TFLOPS``
    docstring for the sustained-MFU calibration.
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
    """RAM in GB. Loose upper bound: weights + optimizer/grads + tokenized text.

    Tokenized text is approximated as ``n_samples x avg_tokens x 4 bytes``
    (BPE/WordPiece token ids fit in int32).

    ``mode``-dependent multiplier on the weights term:
      * ``inference``: 1.3x (weights + intermediate-tensor slack)
      * ``lora``: 1.5x (frozen base + trainable adapters)
      * ``full-finetune`` / anything else: 4.5x (weights + grads + Adam m + v
        + framework slack) — matches the VRAM-side ``4.5W`` accounting so the
        host-pinned optimizer state (Adam mirrors weights) shows up in the
        RAM estimate too. Under-predicting RAM lets a training preset OOM the
        host well before it OOMs the GPU; advisor should upper-bound.
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


def _ram_for_sklearn(
    *,
    stats: DatasetStats,
    embedder_dim: int,
    n_estimators: int,
    max_depth: int,
    n_jobs: int,
) -> float:
    """RandomForest/similar RAM upper bound, aware of ``n_jobs`` replication.

    sklearn spawns ``n_jobs`` worker processes with joblib's loky backend by
    default; each worker holds its own copy of the training feature matrix and
    the trees it grew, so a preset with ``n_jobs=8`` on a 10 k × 1024 embedder
    dataset multiplies the base RAM 8x. Previously sklearn was in
    ``_UNKNOWN_SCORER_MODULES`` and emitted a zero row — classic-heavy's
    real ~2 GB sklearn contribution slipped through invisibly.
    """
    # Per-worker feature matrix (fp64 in sklearn by default).
    per_worker_data = stats.n_samples * embedder_dim * 8
    # Per-worker tree storage: n_estimators × n_leaves × ~32 B/node. Cap
    # n_leaves at n_samples (a tree with max_depth 150 on 10k samples can't
    # actually have 2**150 leaves).
    n_leaves = min(2**max_depth, stats.n_samples) if max_depth > 0 else stats.n_samples
    per_worker_trees = n_estimators * n_leaves * _CATBOOST_BYTES_PER_TREE_NODE
    per_worker = per_worker_data + per_worker_trees
    return float((per_worker * max(1, n_jobs)) / _BYTES_PER_GB)


def _embedder_load_ram_gb(meta: ModelMeta | None) -> float:
    """Extra RAM the process holds when an embedder is loaded — separately from
    any per-driver row that already accounts for it.

    Rationale: classic presets pre-compute embeddings via the embedder, then
    train sklearn/catboost/linear scorers on top. During and after that
    forward pass the process holds: the embedder weights on the compute
    device, a copy in CPU RAM (fp32 from the safetensors load), the tokenizer
    state, HF Trainer buffers, and the cached embeddings themselves. The
    per-driver ``_ram_for_module`` already captures weights x 1.3 for the
    knn/mlknn rows, but the aggregate ``max`` across drivers hides
    contributions from the other classic scorers that are simultaneously in
    memory. This term is added *on top* of the max-driver RAM so classic
    presets like classic-heavy stop under-predicting by ~4x.

    Uses ``total_params × 4`` (fp32) as the weight footprint even when the
    hub reports fp16 storage (``weight_bytes_per_param=2``) — transformers
    up-casts to fp32 at load time by default, so the fp16 disk size
    under-counts real RAM usage by 2x.
    """
    if meta is None:
        return 0.0
    fp32_weights_gb = (meta.total_params * 4) / _BYTES_PER_GB
    # 3.5x factor: raw weights + activation buffers + HF/tokenizer/loader
    # slack. Empirical: real classic-heavy on banking77 (e5-large embedder,
    # sklearn RF n_jobs=8) measured 10.22 GB RAM. At 3.0x we predicted 9.56
    # (1.07x under — still unsafe); at 3.5x we predict 10.85 (0.94x — safely
    # over). Advisor's OOM-avoidance contract requires "err over".
    return fp32_weights_gb * 3.5


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
#
# Small torch models (Kim's TextCNN, LSTM classifier) trained from scratch
# on token ids. The advisor previously emitted a ``not-estimated`` placeholder
# for these — which read as "free/safe" on nn-heavy / nn-medium (predicted 0h
# / 0GB, real 0.3 h + 2.3 GB RAM + 0.7 GB VRAM on banking77).
#
# Cost model: embedding table + head weights (small) + activations that scale
# with batch × seq_len × hidden. We assume a bounded vocabulary (~30k) — one
# HPO trial for TextCNN embeds every token that appears in the training set;
# banking77 tops out around a few thousand unique tokens so 30k is a safe
# upper bound.
_NN_MAX_VOCAB = 30_000
_NN_DEFAULT_SEQ_LEN = 50  # VocabConfig.max_seq_length default
_NN_BYTES_PER_PARAM = 4  # fp32 weights
# fp32 activation storage per (batch, token, hidden) unit, factor absorbs
# ~4x backward overhead + optimizer + gradient state for these tiny models.
_NN_TRAIN_ACT_BYTES_PER_UNIT = 16


def _cnn_param_count(*, embed_dim: int, num_filters: int, n_kernels: int, n_classes: int) -> int:
    """Approximate CNN parameter count: embedding + conv + fc layers."""
    vocab_params = _NN_MAX_VOCAB * embed_dim
    conv_params = num_filters * embed_dim * n_kernels * 5  # avg kernel width ~5
    fc_params = num_filters * n_kernels * max(1, n_classes)
    return vocab_params + conv_params + fc_params


def _rnn_param_count(*, embed_dim: int, hidden_dim: int, n_classes: int) -> int:
    """Approximate LSTM classifier parameter count: embedding + LSTM + fc."""
    vocab_params = _NN_MAX_VOCAB * embed_dim
    # LSTM cell has 4 gates, each with (embed+hidden+1) × hidden params.
    lstm_params = 4 * hidden_dim * (embed_dim + hidden_dim + 1)
    fc_params = hidden_dim * max(1, n_classes)
    return vocab_params + lstm_params + fc_params


def _vram_for_nn(*, params: int, batch_size: int, hidden_dim: int) -> float:
    """Weights + activations for a small torch scorer in training mode.

    Activation term uses ``batch × seq_len × hidden × const`` per the same
    fp32 upper bound as transformers, but with a much smaller effective
    hidden dim (embed_dim/num_filters, not model dim).
    """
    weights_gb = (params * _NN_BYTES_PER_PARAM) / _BYTES_PER_GB
    # Optimizer state (Adam has 2x weights) + gradients (1x weights) = 4x weights total.
    optimizer_gb = 3 * weights_gb
    activations_gb = (
        batch_size * _NN_DEFAULT_SEQ_LEN * hidden_dim * _NN_TRAIN_ACT_BYTES_PER_UNIT
    ) / _BYTES_PER_GB
    return weights_gb + optimizer_gb + activations_gb


def _ram_for_nn(*, params: int, stats: DatasetStats) -> float:
    """CPU-side memory: weights + tokenized text (int32 ids)."""
    weights_gb = (params * _NN_BYTES_PER_PARAM) / _BYTES_PER_GB
    tokens_gb = (stats.n_samples * _NN_DEFAULT_SEQ_LEN * 4) / _BYTES_PER_GB
    return weights_gb + tokens_gb


def _time_for_nn(
    *,
    n_trials: int,
    epochs: int,
    batch_size: int,
    n_samples: int,
    params_millions: float,
    device_class: str,
) -> float:
    """Reuse the transformer FLOPs formula for a small torch model.

    Small models are memory-bandwidth-bound, not compute-bound, so the FLOPs
    formula slightly *under*-predicts wall-time. Empirically for banking77
    nn-heavy we measured 0.32 h across 55 trials — the formula lands within
    2x of that, which is enough for a cost-ranking estimate.
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
