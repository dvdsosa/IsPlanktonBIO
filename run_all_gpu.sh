#!/bin/bash
#
# run_all_gpu.sh [runs] [backends]
#
# Desktop-GPU sweep: benchmarks the IsPlanktonBIO pipeline with selected Stage-2
# segmentation backends by calling benchmark_gpu.sh for each, collecting FPS +
# consumption (GPU / CPU / RAM). Desktop counterpart of the Kria `run_all_*`
# wrappers, reduced to a single orchestrator.
#
#   [runs]      repetitions per backend (default 10)
#   [backends]  which backends to run: "opencv", "sam2", or "both" (default "both")
#
# It loads the test set into RAM (/dev/shm) if missing, as the README requires,
# and removes it afterwards only if this script created it.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="${1:-10}"
BACKENDS_ARG="${2:-both}"

DATASET_SRC="${DATASET_SRC:-/home/dsosatr/tesis/DYB-VOS/test}"
RAM_DIR="${RAM_DIR:-/dev/shm/test_dataset_ram}"

# Parse backends parameter
case "$BACKENDS_ARG" in
    both)
        METHODS=(opencv sam2)
        ;;
    opencv)
        METHODS=(opencv)
        ;;
    sam2)
        METHODS=(sam2)
        ;;
    *)
        echo "ERROR: backends must be 'opencv', 'sam2', or 'both' (got '$BACKENDS_ARG')" >&2
        exit 1
        ;;
esac

# ----------------------------------------------------------------------------
# Load dataset into RAM (README: apples-to-apples with the Kria RAM-backed FS)
# ----------------------------------------------------------------------------
created_ram=0
if [ ! -d "$RAM_DIR" ]; then
    if [ -d "$DATASET_SRC" ]; then
        echo "Loading test set into RAM: $DATASET_SRC -> $RAM_DIR"
        cp -r "$DATASET_SRC" "$RAM_DIR" && created_ram=1
    else
        echo "ERROR: neither $RAM_DIR nor $DATASET_SRC exist. Set DATASET_SRC/RAM_DIR." >&2
        exit 1
    fi
fi

cleanup() {
    [ "$created_ram" = 1 ] && { echo "Freeing RAM dataset: $RAM_DIR"; rm -rf "$RAM_DIR"; }
}

cleanup_on_interrupt() {
    echo ""
    echo "=========================================="
    echo "Interrupt detected — killing subprocesses"
    echo "=========================================="
    # Kill benchmark_gpu.sh and any python processes it spawned
    pkill -P $$ 2>/dev/null || true
    # Also kill any lingering python processes from the pipeline
    pkill -f "python -m src.main" 2>/dev/null || true
    # Clean up lock files
    rm -f "$SCRIPT_DIR/output/.benchmark_gpu.lock" /tmp/.benchmark_gpu.lock 2>/dev/null || true
    cleanup
    exit 130  # standard exit code for SIGINT
}

trap cleanup EXIT
trap cleanup_on_interrupt INT TERM

# ----------------------------------------------------------------------------
# Run each backend
# ----------------------------------------------------------------------------
PASSED=(); FAILED=()
for method in "${METHODS[@]}"; do
    echo ""
    echo "###################################################################"
    echo "# Backend: $method  ($RUNS runs)  -  $(date)"
    echo "###################################################################"
    if bash "$SCRIPT_DIR/benchmark_gpu.sh" "$method" "$RUNS"; then
        PASSED+=("$method")
    else
        echo "  [FAIL] $method (exit=$?)"
        FAILED+=("$method")
    fi
done

# ----------------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------------
echo ""
echo "==================================================================="
echo "SWEEP DONE - $(date)"
echo "  Passed (${#PASSED[@]}): ${PASSED[*]:-none}"
echo "  Failed (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "==================================================================="
for method in "${PASSED[@]}"; do
    echo ""
    cat "$SCRIPT_DIR/output/metrics/metrics_gpu_${method}.txt" 2>/dev/null
done
