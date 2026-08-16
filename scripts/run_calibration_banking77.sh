#!/usr/bin/env bash
# Run the advisor calibration across every bundled preset on DeepPavlov/banking77.
#
# WARNING: transformers-heavy on banking77 (10k train samples, 77 classes) can
# take *many* hours on a single GPU. Set MAX_TRIALS to a small number for a
# fast sanity check, or leave it unset to let each preset use its bundled
# ``hpo_config.n_trials``.
#
# Environment overrides:
#   DATASET          HF Hub repo id (default: DeepPavlov/banking77)
#   DATASETS         Space-separated list of dataset ids (default: unset -> use
#                    DATASET). Every preset runs against every dataset — useful
#                    to sweep a small + large + multilabel + long-token shape in
#                    a single invocation.
#   SUBSAMPLE_PER_CLASS  Cap each class to N training samples (deterministic
#                        first-N slice) before running — turns banking77 into a
#                        small dataset without needing a separate corpus.
#   REPEATS          Run each (preset, dataset) N times. The summary prints
#                    mean ± stdev of actual measurements across repeats so
#                    small ratio gaps become judgeable. Default: 1.
#   PRESETS          Space-separated preset names OR paths (default: every bundled preset).
#                    Items ending in .yaml/.yml are treated as file paths — use
#                    scripts/coverage_preset.yaml to exercise lora/ptuning/dnnc/gcn/
#                    cross-encoder in one small run without touching bundled presets.
#   MAX_TRIALS       Cap for hpo_config.n_trials (default: unset -> preset default)
#   WANDB            If non-empty, pass --wandb so system metrics land in wandb.ai
#   RUN_NAME         Suffix appended to each preset's LoggingConfig.run_name — the
#                    resulting name is ``{preset}_{RUN_NAME}`` (default: unset ->
#                    autointent generates a random name)
#   OUTPUT_DIR       Where JSON reports + logs land (default: ./calibration_runs)
#   SKIP_FIT         If non-empty, only run preflight (no real fit) — fast sanity check
#   THREADS_PER_JOB  Cap for BLAS/OpenMP/torch intra-op threads per HPO trial
#                    (default: 1). Increase carefully — sklearn's own ``n_jobs`` and
#                    HPO parallelism multiply on top, so oversubscription is easy
#                    on many-core boxes.
#   COLD             If non-empty, pass --clear-embedding-cache so each preset
#                    starts with an empty embeddings cache (measure COLD cost).
#   BUDGET_VRAM_GB   Force run_preflight to see a specific VRAM budget instead
#                    of what the box exposes — lets you exercise severity paths
#                    (red/yellow/green + findings_over) on a big box.
#   REQUIRE_CUDA     If non-empty, pass --require-cuda so the run fails fast when
#                    torch.cuda.is_available() is False (guards against silent
#                    CPU-fallback caused by CUDA-driver / torch-wheel mismatch).
#
# Examples:
#   scripts/run_calibration_banking77.sh                           # full sweep, serial
#   MAX_TRIALS=3 scripts/run_calibration_banking77.sh              # quick sweep
#   PRESETS="classic-light nn-medium" scripts/run_calibration_banking77.sh
#   WANDB=1 MAX_TRIALS=5 scripts/run_calibration_banking77.sh
#   RUN_NAME=calib_2026_07 WANDB=1 scripts/run_calibration_banking77.sh
#   THREADS_PER_JOB=4 scripts/run_calibration_banking77.sh         # 4-thread BLAS
#   SUBSAMPLE_PER_CLASS=5 scripts/run_calibration_banking77.sh     # small-dataset shape
#   DATASETS="DeepPavlov/banking77 DeepPavlov/clinc150" scripts/run_calibration_banking77.sh
#   REPEATS=3 MAX_TRIALS=3 scripts/run_calibration_banking77.sh    # variance bars
#   PRESETS="scripts/coverage_preset.yaml" MAX_TRIALS=1 scripts/run_calibration_banking77.sh
#       # exercise lora/ptuning/dnnc/gcn/description_cross — modules no bundled preset touches

set -euo pipefail

# Resolve repo root even when the script is called from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-DeepPavlov/banking77}"
# DATASETS overrides DATASET when set; enables multi-dataset sweeps.
if [[ -n "${DATASETS:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word-split from env
    DATASET_ARR=($DATASETS)
else
    DATASET_ARR=("$DATASET")
fi
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/calibration_runs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_JSON="$OUTPUT_DIR/banking77_$TIMESTAMP.json"
LOG_FILE="$OUTPUT_DIR/banking77_$TIMESTAMP.log"

export WANDB_PROJECT="autointent_feasibility"

# ---------------------------------------------------------------------------
# CPU thread caps — set BEFORE python starts, because numpy/torch/sklearn read
# them at import time. Without these, on a 16+ core box each BLAS-backed
# operation defaults to N-thread pools which multiply with sklearn's own
# ``n_jobs`` and HPO parallelism → the machine oversubscribes and stalls.
# ---------------------------------------------------------------------------
THREADS_PER_JOB="${THREADS_PER_JOB:-1}"
export OMP_NUM_THREADS="$THREADS_PER_JOB"
export MKL_NUM_THREADS="$THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$THREADS_PER_JOB"
# HF tokenizers deadlock on fork if left in parallel mode.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# torch reads OMP_NUM_THREADS for intra-op, but set explicitly too — belt-and-braces.
export PYTORCH_NUM_THREADS="$THREADS_PER_JOB"

mkdir -p "$OUTPUT_DIR"

# Assemble optional flags.
EXTRA_FLAGS=()
if [[ -n "${MAX_TRIALS:-}" ]]; then
    EXTRA_FLAGS+=("--max-trials" "$MAX_TRIALS")
fi
if [[ -n "${WANDB:-}" ]]; then
    EXTRA_FLAGS+=("--wandb")
fi
if [[ -n "${SKIP_FIT:-}" ]]; then
    EXTRA_FLAGS+=("--skip-fit")
fi
if [[ -n "${RUN_NAME:-}" ]]; then
    EXTRA_FLAGS+=("--run-name" "$RUN_NAME")
fi
if [[ -n "${COLD:-}" ]]; then
    EXTRA_FLAGS+=("--clear-embedding-cache")
fi
if [[ -n "${BUDGET_VRAM_GB:-}" ]]; then
    EXTRA_FLAGS+=("--budget-vram-gb" "$BUDGET_VRAM_GB")
fi
if [[ -n "${REQUIRE_CUDA:-}" ]]; then
    EXTRA_FLAGS+=("--require-cuda")
fi
if [[ -n "${SUBSAMPLE_PER_CLASS:-}" ]]; then
    EXTRA_FLAGS+=("--subsample-per-class" "$SUBSAMPLE_PER_CLASS")
fi
if [[ -n "${REPEATS:-}" ]]; then
    EXTRA_FLAGS+=("--repeats" "$REPEATS")
fi

# Preset list: pull it from the advisor package at runtime unless overridden,
# so the script auto-discovers presets that are added later.
if [[ -n "${PRESETS:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word-split from env
    PRESET_ARR=($PRESETS)
else
    PRESET_ARR=()
    while IFS= read -r preset; do
        PRESET_ARR+=("$preset")
    done < <(
# Must go through `uv run` — the bare interpreter has no autointent on its
# path, which made the no-PRESETS default invocation die with an empty list.
uv run --no-sync python - <<'PY'
from autointent._advisor import BUNDLED_PRESETS
for name in BUNDLED_PRESETS:
    print(name)
PY
    )
fi

echo "Repo:            $REPO_ROOT"
echo "Datasets:        ${DATASET_ARR[*]}"
echo "Presets:         ${PRESET_ARR[*]}"
echo "Output:          $OUTPUT_JSON"
echo "Log:             $LOG_FILE"
echo "Flags:           ${EXTRA_FLAGS[*]:-<none>}"
echo "Threads per job: $THREADS_PER_JOB (OMP/MKL/OpenBLAS/torch)"
echo

uv run --no-sync python scripts/calibrate_advisor.py \
    --dataset "${DATASET_ARR[@]}" \
    --presets "${PRESET_ARR[@]}" \
    --output "$OUTPUT_JSON" \
    "${EXTRA_FLAGS[@]}" \
    2>&1 | tee "$LOG_FILE"

echo
echo "Done. JSON: $OUTPUT_JSON"
echo "         Log:  $LOG_FILE"
