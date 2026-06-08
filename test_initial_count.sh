#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-test_initial_count_results}"
INITIAL_COUNTS="${INITIAL_COUNTS:-10 50 100 all}"

MODEL_LAYERS_NUM="${MODEL_LAYERS_NUM:-10}"
MODEL_LAYERS_WIDTH="${MODEL_LAYERS_WIDTH:-64}"
QUASI_SPACE="${QUASI_SPACE:-2000}"
BOUNDARY_COUNT="${BOUNDARY_COUNT:-200}"
COLLOCATION_COUNT="${COLLOCATION_COUNT:-10000}"
ADAM_EPOCHS="${ADAM_EPOCHS:-2000}"
LBFGS_ITERS="${LBFGS_ITERS:-2000}"
ALTERNATIVE_COLLOCATION_COUNT="${ALTERNATIVE_COLLOCATION_COUNT:-$COLLOCATION_COUNT}"
ALTERNATIVE_ADAM_EPOCHS="${ALTERNATIVE_ADAM_EPOCHS:-$ADAM_EPOCHS}"
ALTERNATIVE_LBFGS_ITERS="${ALTERNATIVE_LBFGS_ITERS:-$LBFGS_ITERS}"
STEPS="${STEPS:-3}"
N_START="${N_START:-100}"
SEED="${SEED:-42}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export MPLCONFIGDIR

mkdir -p "$TEST_OUTPUT_DIR"

for initial_count in $INITIAL_COUNTS; do
    output="$TEST_OUTPUT_DIR/initial_${initial_count}.hdf5"
    if [[ -f "$output" ]]; then
        echo "Skipping existing $output"
        continue
    fi

    echo "Generating $output: INITIAL_COUNT=$initial_count"
    "$PYTHON" main.py generate \
        --data-output "$output" \
        --quasi-space "$QUASI_SPACE" \
        --model-layers-num "$MODEL_LAYERS_NUM" \
        --model-layers-width "$MODEL_LAYERS_WIDTH" \
        --initial-count "$initial_count" \
        --boundary-count "$BOUNDARY_COUNT" \
        --collocation-count "$COLLOCATION_COUNT" \
        --adam-epochs "$ADAM_EPOCHS" \
        --lbfgs-iters "$LBFGS_ITERS" \
        --alternative-collocation-count "$ALTERNATIVE_COLLOCATION_COUNT" \
        --alternative-adam-epochs "$ALTERNATIVE_ADAM_EPOCHS" \
        --alternative-lbfgs-iters "$ALTERNATIVE_LBFGS_ITERS" \
        --steps "$STEPS" \
        --n-start "$N_START" \
        --seed "$SEED"
done
