#!/usr/bin/env bash
# Run the advisor calibration across every bundled preset on DeepPavlov/banking77.
#
# WARNING: transformers-heavy on banking77 (10k train samples, 77 classes) can
# take *many* hours on a single GPU. Set MAX_TRIALS to a small number for a
# fast sanity check, or leave it unset to let each preset use its bundled
# ``hpo_config.n_trials``.
#
# Environment overrides:
#   DATASET      HF Hub repo id (default: DeepPavlov/banking77)
#   PRESETS      Space-separated preset names (default: every bundled preset)
#   MAX_TRIALS   Cap for hpo_config.n_trials (default: unset -> preset default)
#   WANDB        If non-empty, pass --wandb so system metrics land in wandb.ai
#   OUTPUT_DIR   Where JSON reports + logs land (default: ./calibration_runs)
#   SKIP_FIT     If non-empty, only run preflight (no real fit) for a fast sanity check
#
# Examples:
#   scripts/run_calibration_banking77.sh                           # full sweep
#   MAX_TRIALS=3 scripts/run_calibration_banking77.sh              # quick sweep
#   PRESETS="classic-light nn-medium" scripts/run_calibration_banking77.sh
#   WANDB=1 MAX_TRIALS=5 scripts/run_calibration_banking77.sh

set -euo pipefail

# Resolve repo root even when the script is called from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-DeepPavlov/banking77}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/calibration_runs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_JSON="$OUTPUT_DIR/banking77_$TIMESTAMP.json"
LOG_FILE="$OUTPUT_DIR/banking77_$TIMESTAMP.log"

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
python - <<'PY'
from autointent._advisor import BUNDLED_PRESETS
for name in BUNDLED_PRESETS:
    print(name)
PY
    )
fi

echo "Repo:      $REPO_ROOT"
echo "Dataset:   $DATASET"
echo "Presets:   ${PRESET_ARR[*]}"
echo "Output:    $OUTPUT_JSON"
echo "Log:       $LOG_FILE"
echo "Flags:     ${EXTRA_FLAGS[*]:-<none>}"
echo

uv run --no-sync python scripts/calibrate_advisor.py \
    --dataset "$DATASET" \
    --presets "${PRESET_ARR[@]}" \
    --output "$OUTPUT_JSON" \
    "${EXTRA_FLAGS[@]}" \
    2>&1 | tee "$LOG_FILE"

echo
echo "Done. JSON: $OUTPUT_JSON"
echo "         Log:  $LOG_FILE"
