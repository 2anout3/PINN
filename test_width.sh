#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-test_width_results}"
WIDTHS="${WIDTHS:-2 4 8 16 24 32 40 64}"

MODEL_LAYERS_NUM="${MODEL_LAYERS_NUM:-10}"
QUASI_SPACE="${QUASI_SPACE:-2000}"
INITIAL_COUNT="${INITIAL_COUNT:-all}"
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

for width in $WIDTHS; do
    output="$TEST_OUTPUT_DIR/layer_width_${width}.hdf5"
    if [[ -f "$output" ]]; then
        echo "Skipping existing $output"
        continue
    fi

    echo "Generating $output: MODEL_LAYERS_WIDTH=$width"
    "$PYTHON" main.py generate \
        --data-output "$output" \
        --quasi-space "$QUASI_SPACE" \
        --model-layers-num "$MODEL_LAYERS_NUM" \
        --model-layers-width "$width" \
        --initial-count "$INITIAL_COUNT" \
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
