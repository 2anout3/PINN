#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-test_results}"
EPOCHS="${EPOCHS:-0 100 200 500 1000 2000}"

MODEL_LAYERS_NUM="${MODEL_LAYERS_NUM:-10}"
MODEL_LAYERS_WIDTH="${MODEL_LAYERS_WIDTH:-64}"
QUASI_SPACE="${QUASI_SPACE:-2000}"
INITIAL_COUNT="${INITIAL_COUNT:-all}"
BOUNDARY_COUNT="${BOUNDARY_COUNT:-200}"
COLLOCATION_COUNT="${COLLOCATION_COUNT:-10000}"
ALTERNATIVE_COLLOCATION_COUNT="${ALTERNATIVE_COLLOCATION_COUNT:-$COLLOCATION_COUNT}"
STEPS="${STEPS:-3}"
N_START="${N_START:-100}"
SEED="${SEED:-42}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export MPLCONFIGDIR

mkdir -p "$TEST_OUTPUT_DIR"

run_generate() {
    local output="$1"
    local adam_epochs="$2"
    local lbfgs_iters="$3"

    if [[ -f "$output" ]]; then
        echo "Skipping existing $output"
        return
    fi

    echo "Generating $output: ADAM_EPOCHS=$adam_epochs, LBFGS_ITERS=$lbfgs_iters"
    "$PYTHON" main.py generate \
        --data-output "$output" \
        --quasi-space "$QUASI_SPACE" \
        --model-layers-num "$MODEL_LAYERS_NUM" \
        --model-layers-width "$MODEL_LAYERS_WIDTH" \
        --initial-count "$INITIAL_COUNT" \
        --boundary-count "$BOUNDARY_COUNT" \
        --collocation-count "$COLLOCATION_COUNT" \
        --adam-epochs "$adam_epochs" \
        --lbfgs-iters "$lbfgs_iters" \
        --alternative-collocation-count "$ALTERNATIVE_COLLOCATION_COUNT" \
        --alternative-adam-epochs "$adam_epochs" \
        --alternative-lbfgs-iters "$lbfgs_iters" \
        --steps "$STEPS" \
        --n-start "$N_START" \
        --seed "$SEED"
}

for adam_epochs in $EPOCHS; do
    run_generate "$TEST_OUTPUT_DIR/adam_${adam_epochs}.hdf5" "$adam_epochs" 0
done

for lbfgs_iters in $EPOCHS; do
    run_generate "$TEST_OUTPUT_DIR/lbfgs_${lbfgs_iters}.hdf5" 0 "$lbfgs_iters"
done
