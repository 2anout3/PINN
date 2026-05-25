#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-generate}"
if [[ "$MODE" == "-h" || "$MODE" == "--help" || "$MODE" == "help" ]]; then
    MODE="help"
    shift || true
elif [[ "$MODE" == "generate" || "$MODE" == "visualize" || "$MODE" == "all" ]]; then
    shift || true
else
    MODE="generate"
fi

PYTHON="${PYTHON:-python}"
DATA_OUTPUT="${DATA_OUTPUT:-computationData.hdf5}"
MODEL_LAYERS_NUM="${MODEL_LAYERS_NUM:-10}"
MODEL_LAYERS_WIDTH="${MODEL_LAYERS_WIDTH:-64}"
QUASI_SPACE="${QUASI_SPACE:-2000}"
INITIAL_COUNT="${INITIAL_COUNT:-all}"
BOUNDARY_COUNT="${BOUNDARY_COUNT:-200}"
COLLOCATION_COUNT="${COLLOCATION_COUNT:-10000}"
ADAM_EPOCHS="${ADAM_EPOCHS:-2000}"
LBFGS_ITERS="${LBFGS_ITERS:-2000}"
ALTERNATIVE_COLLOCATION_COUNT="${ALTERNATIVE_COLLOCATION_COUNT:-$COLLOCATION_COUNT}"
ALTERNATIVE_ADAM_EPOCHS="${ALTERNATIVE_ADAM_EPOCHS:-2000}"
ALTERNATIVE_LBFGS_ITERS="${ALTERNATIVE_LBFGS_ITERS:-$LBFGS_ITERS}"
STEPS="${STEPS:-3}"
N_START="${N_START:-100}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export MPLCONFIGDIR

usage() {
    cat <<EOF
Usage:
  ./start.sh generate [extra generate args]
  ./start.sh visualize [extra visualize args]
  ./start.sh all

Environment overrides:
  PYTHON=$PYTHON
  DATA_OUTPUT=$DATA_OUTPUT
  MODEL_LAYERS_NUM=$MODEL_LAYERS_NUM
  MODEL_LAYERS_WIDTH=$MODLE_LAYERS_WIDTH
  QUASI_SPACE=$QUASI_SPACE
  INITIAL_COUNT=$INITIAL_COUNT
  BOUNDARY_COUNT=$BOUNDARY_COUNT
  COLLOCATION_COUNT=$COLLOCATION_COUNT
  ADAM_EPOCHS=$ADAM_EPOCHS
  LBFGS_ITERS=$LBFGS_ITERS
  ALTERNATIVE_COLLOCATION_COUNT=$ALTERNATIVE_COLLOCATION_COUNT
  ALTERNATIVE_ADAM_EPOCHS=$ALTERNATIVE_ADAM_EPOCHS
  ALTERNATIVE_LBFGS_ITERS=$ALTERNATIVE_LBFGS_ITERS
  STEPS=$STEPS
  N_START=$N_START
EOF
}

generate() {
    "$PYTHON" main.py generate \
        --data-output "$DATA_OUTPUT" \
        --quasi-space "$QUASI_SPACE" \
        --model-layers-num "$MODEL_LAYERS_NUM" \
        --model-layers-width "$MODEL_LAYERS_WIDTH" \
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
        "$@"
}

visualize() {
    "$PYTHON" main.py visualize \
        --data-input "$DATA_OUTPUT" \
        "$@"
}

case "$MODE" in
    generate)
        generate "$@"
        ;;
    visualize)
        visualize "$@"
        ;;
    all)
        generate
        visualize "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
