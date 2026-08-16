#!/usr/bin/env bash
# Phase 2 of issue #39, one preset per PROCESS.
#
# The sweep driver (run_calibration_banking77.sh) walks every preset inside a
# single python process. That is fine on an A100 and actively misleading on a
# 6 GB card: the first preset to OOM leaves its model + optimizer state + HPO
# trial objects alive, so the next preset starts several GB in the hole and
# OOMs on an allocation it would never have made on a clean device. Observed
# directly on this box — `classic-light` (predicted 1.8 GB, AMPLE) "OOMed"
# only because `transformers-no-hpo` had leaked 5.4 GB immediately before it.
#
# Process isolation is the only airtight fix: the kernel reclaims the GPU on
# exit no matter what the python object graph is holding.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/calibration_runs/phase2_isolated}"
mkdir -p "$OUTPUT_DIR"

# Cheap/AMPLE presets first, the expected-OOM ones last.
PRESETS="${PRESETS:-classic-light zero-shot-encoders transformers-no-hpo transformers-heavy}"

for preset in $PRESETS; do
    echo "############ $preset ############"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    COLD=1 REQUIRE_CUDA=1 MAX_TRIALS="${MAX_TRIALS:-2}" \
        SUBSAMPLE_PER_CLASS="${SUBSAMPLE_PER_CLASS:-30}" \
        PRESETS="$preset" RUN_NAME="${RUN_NAME:-laptop6gb_fit}" \
        OUTPUT_DIR="$OUTPUT_DIR/$preset" \
        scripts/run_calibration_banking77.sh || echo "!!! $preset driver exited non-zero"
    echo "############ $preset done; GPU after process exit:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
done

echo "All presets done. JSONs under $OUTPUT_DIR"
