#!/bin/bash
#
# benchmark_gpu.sh <method> [runs] [tag]
#
# Runs the IsPlanktonBIO pipeline (README: `python -m src.main --config ...`)
# N times on the GPU PC while sampling resource usage, then reports FPS and
# consumption as mean +/- std (sample, N-1). This is the desktop-GPU counterpart
# of the Kria `resnet50-power-memory.sh` + `monitor_power.sh` pair, merged into
# a single self-contained script.
#
#   <method>  segmentation backend: 'opencv' (default) or 'sam2'
#   [runs]    number of repetitions (default 10)
#   [tag]     label for output files (default = <method>)
#
# Per run it captures, as best the desktop allows:
#   - Average FPS                 (images / total elapsed time from src.main)
#   - Peak process RAM  (VmHWM)   -> /proc/<pid>/status
#   - Avg / peak GPU power (W)     -> nvidia-smi power.draw
#   - Peak GPU memory (MiB)        -> nvidia-smi memory.used
#   - Avg GPU utilization (%)      -> nvidia-smi utilization.gpu
#   - Avg CPU utilization (%)      -> /proc/stat deltas
#   - Avg CPU package power (W)    -> Intel/AMD RAPL (needs root; N/A otherwise)
#   - Avg CPU temperature (C)      -> `sensors` Tctl (k10temp / coretemp)
#
# Outputs:
#   output/metrics/metrics_gpu_<tag>.txt   (this run's summary, also echoed)
#   output/inference_results_gpu_<tag>.json
#   output/species_results_gpu_<tag>.json
#
set -uo pipefail

# ----------------------------------------------------------------------------
# Args & paths
# ----------------------------------------------------------------------------
METHOD="${1:-opencv}"
NUM_RUNS="${2:-10}"
TAG="${3:-$METHOD}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
CONFIG_SRC="$BASE_DIR/configs/default_config.yaml"
OUTPUT_DIR="$BASE_DIR/output"
METRICS_DIR="$OUTPUT_DIR/metrics"
GPU_INDEX="${GPU_INDEX:-0}"
RAPL_ENERGY="/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX_FILE="/sys/class/powercap/intel-rapl:0/max_energy_range_uj"
# Wrap point of the RAPL energy counter (microjoules). It is a bounded counter
# that resets to 0 at this value, so a run can straddle the boundary. Used to
# correct negative deltas. Fallback keeps arithmetic valid if the file is absent.
RAPL_MAX="$(cat "$RAPL_MAX_FILE" 2>/dev/null || echo 0)"

mkdir -p "$METRICS_DIR"

if [[ "$METHOD" != "opencv" && "$METHOD" != "sam2" ]]; then
    echo "ERROR: method must be 'opencv' or 'sam2' (got '$METHOD')" >&2
    exit 1
fi

# Single-instance lock: every run reads/writes the same output/*.json files, so
# two concurrent benchmarks would corrupt each other's results (and hit
# PermissionError if one is root). flock auto-releases when the script exits.
LOCK_FILE="$OUTPUT_DIR/.benchmark_gpu.lock"
mkdir -p "$OUTPUT_DIR"
exec 9>"$LOCK_FILE" 2>/dev/null || exec 9>"/tmp/.benchmark_gpu.lock"
if ! flock -n 9; then
    echo "ERROR: another benchmark_gpu.sh is already running (lock: $LOCK_FILE)." >&2
    echo "       Wait for it to finish, or stop it, before starting a new one." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Python environment (README uses the conda env 'paper2')
# ----------------------------------------------------------------------------
if ! python -c "import torch" >/dev/null 2>&1; then
    if [ -f /home/dsosatr/miniconda3/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        source /home/dsosatr/miniconda3/etc/profile.d/conda.sh
        conda activate "${CONDA_ENV:-paper2}" 2>/dev/null || true
    fi
fi
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: could not import torch. Activate the env first: conda activate paper2" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Build a config for the requested backend (only segmentation.method changes)
# ----------------------------------------------------------------------------
RUN_CONFIG="$METRICS_DIR/config_gpu_${TAG}.yaml"
sed -E "s/^([[:space:]]*method:).*/\1 '${METHOD}'/" "$CONFIG_SRC" > "$RUN_CONFIG"
echo "Backend '$METHOD' -> $(grep -E '^[[:space:]]*method:' "$RUN_CONFIG" | head -1 | sed 's/^ *//')"

# Dataset must be preloaded into RAM (see README). Fail early with a clear hint.
DATASET_ROOT="$(grep -E '^[[:space:]]*dataset_root_path:' "$RUN_CONFIG" | head -1 | sed -E "s/.*:[[:space:]]*['\"]?//; s/['\"].*//")"
if [ -n "$DATASET_ROOT" ] && [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: dataset_root_path '$DATASET_ROOT' not found." >&2
    echo "       Load the test set into RAM first (README):" >&2
    echo "       cp -r /home/dsosatr/tesis/DYB-VOS/test /dev/shm/test_dataset_ram" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
# mean+std (sample, N-1) from values on stdin -> "mean std"
stats() {
    awk '{v[NR]=$1; s+=$1}
         END{ if(NR==0){print "0 0"; exit}
              m=s/NR;
              if(NR>1){for(i=1;i<=NR;i++){d=v[i]-m; ss+=d*d} sd=sqrt(ss/(NR-1))} else sd=0;
              printf "%.4f %.4f", m, sd }'
}
read_rapl()      { cat "$RAPL_ENERGY" 2>/dev/null; }          # empty if not permitted
read_cpu_temp()  { sensors 2>/dev/null | awk '/Tctl:|Package id 0:/{gsub(/[+°C]/,"",$NF); print $NF; exit}'; }
cpu_snapshot()   { read -r _ a b c d e f g h _ < /proc/stat; echo "$((a+b+c+d+e+f+g+h)) $((d+e))"; }  # total idle

# per-run result arrays
fps_v=(); rss_v=(); gpow_v=(); gpeak_v=(); gmem_v=(); gutil_v=(); cutil_v=(); cpow_v=(); ctemp_v=()
gjimg_v=(); cjimg_v=(); tjimg_v=()   # energy per inference (J/image): GPU, CPU, total

echo "==================================================================="
echo "Benchmark: backend=$METHOD | runs=$NUM_RUNS | GPU#$GPU_INDEX | $(date)"
echo "==================================================================="

for ((i=1; i<=NUM_RUNS; i++)); do
    echo "-----------------------------------"
    echo "Run #$i/$NUM_RUNS"

    gpu_samples="$(mktemp)"
    peak_rss_kb=0
    ctemp_sum=0; ctemp_n=0

    # Baselines for whole-run CPU util (/proc/stat) and CPU energy (RAPL).
    # RAPL energy is accumulated per-sample (not as a single start/end delta) so
    # any number of counter wraps during a long run is handled exactly.
    read -r cpu_total0 cpu_idle0 < <(cpu_snapshot)
    rapl_prev="$(read_rapl)"
    cpu_uj_acc=0
    t0="$(date +%s.%N)"

    # Start each run from a clean slate. The pipeline rewrites these files (the
    # areas JSON on every image, the results JSON at the end); a stale copy left
    # root-owned by a previous `sudo` run would otherwise raise PermissionError
    # mid-run. output/ is user-owned, so a foreign-owned file inside it is still
    # deletable.
    rm -f "$OUTPUT_DIR/species_results_areas.json" "$OUTPUT_DIR/inference_results.json"

    # Launch the pipeline exactly as documented in the README.
    # PYTHONUNBUFFERED=1 guarantees the final summary lines are flushed to the
    # log; each run keeps its own log so a parse failure is always inspectable.
    run_log="$METRICS_DIR/run_${TAG}_${i}.log"
    PYTHONUNBUFFERED=1 python -m src.main --config "$RUN_CONFIG" > "$run_log" 2>&1 &
    CMD_PID=$!

    # Sample while the process is alive (~5 Hz).
    while kill -0 "$CMD_PID" 2>/dev/null; do
        hwm=$(grep VmHWM "/proc/$CMD_PID/status" 2>/dev/null | awk '{print $2}')
        [ -n "$hwm" ] && [ "$hwm" -gt "$peak_rss_kb" ] && peak_rss_kb=$hwm

        nvidia-smi -i "$GPU_INDEX" \
            --query-gpu=power.draw,utilization.gpu,memory.used \
            --format=csv,noheader,nounits 2>/dev/null \
            | tr -d ' ' | awk -F',' 'NF>=3{print $1, $2, $3}' >> "$gpu_samples"

        tc="$(read_cpu_temp)"
        if [ -n "$tc" ]; then ctemp_sum=$(echo "$ctemp_sum + $tc" | bc -l); ctemp_n=$((ctemp_n+1)); fi

        # Accumulate CPU package energy from RAPL, correcting counter wrap.
        if [ -n "$rapl_prev" ]; then
            r_now="$(read_rapl)"
            if [ -n "$r_now" ]; then
                d=$((r_now - rapl_prev))
                [ "$d" -lt 0 ] && d=$((d + RAPL_MAX))   # counter wrapped since last sample
                cpu_uj_acc=$((cpu_uj_acc + d))
                rapl_prev="$r_now"
            fi
        fi

        sleep 0.2
    done
    wait "$CMD_PID"

    t1="$(date +%s.%N)"
    read -r cpu_total1 cpu_idle1 < <(cpu_snapshot)

    # --- FPS from src.main output ---
    # Extract with regex (not positional awk): tqdm's final progress bar is
    # written to stderr without a trailing newline and can glue itself onto the
    # summary line, which would shift awk field positions. grep -oE is immune.
    n_img=$(grep -oE "Total number of images in test set: [0-9]+" "$run_log" | grep -oE "[0-9]+$" | tail -1)
    elapsed=$(grep -oE "Total elapsed time: [0-9.]+" "$run_log" | grep -oE "[0-9.]+" | tail -1)
    if [ -z "$elapsed" ]; then
        # Fallback: derive elapsed from mean processing time per image.
        mt=$(grep -oE "Mean processing time per image: [0-9.]+" "$run_log" | grep -oE "[0-9.]+" | tail -1)
        [ -n "$mt" ] && [ -n "$n_img" ] && elapsed=$(echo "$n_img * $mt" | bc -l)
    fi
    if [ -n "$n_img" ] && [ -n "$elapsed" ] && [ "$(echo "$elapsed > 0" | bc -l)" -eq 1 ]; then
        fps=$(echo "scale=4; $n_img / $elapsed" | bc -l)
        rm -f "$run_log"
    else
        fps=0
        echo "  WARN: no FPS parsed in run #$i. Kept log; last lines of $run_log:"
        tail -n 15 "$run_log" | sed 's/^/      | /'
    fi

    # --- GPU aggregates (avg power, peak power, peak mem, avg util) ---
    read -r gpu_pavg gpu_ppeak gpu_mpeak gpu_uavg < <(
        awk '{ps+=$1; if($1>pp)pp=$1; if($3>mm)mm=$3; us+=$2; n++}
             END{ if(n==0){print "0 0 0 0"} else printf "%.2f %.2f %.0f %.2f", ps/n, pp, mm, us/n }' "$gpu_samples")

    # --- CPU utilization over the whole run ---
    d_total=$((cpu_total1 - cpu_total0)); d_idle=$((cpu_idle1 - cpu_idle0))
    if [ "$d_total" -gt 0 ]; then
        cpu_util=$(echo "scale=2; 100 * ($d_total - $d_idle) / $d_total" | bc -l)
    else cpu_util=0; fi

    # --- CPU package power + energy via RAPL (needs root; N/A otherwise) ---
    # Energy is the exact accumulated counter delta; avg power = energy / time.
    run_secs=$(echo "$t1 - $t0" | bc -l)
    if [ "$cpu_uj_acc" -gt 0 ] && [ "$(echo "$run_secs > 0" | bc -l)" -eq 1 ]; then
        cpu_energy=$(echo "scale=2; $cpu_uj_acc / 1000000" | bc -l)   # joules for this run
        cpu_pow=$(echo "scale=2; $cpu_energy / $run_secs" | bc -l)    # avg watts
    else cpu_pow="NA"; cpu_energy="NA"; fi

    # --- CPU temperature (avg of samples) ---
    if [ "$ctemp_n" -gt 0 ]; then cpu_temp=$(echo "scale=1; $ctemp_sum / $ctemp_n" | bc -l); else cpu_temp="NA"; fi

    peak_rss_mb=$(echo "scale=2; $peak_rss_kb / 1024" | bc)
    rm -f "$gpu_samples"

    # --- Energy per inference (J/image) = avg_power * elapsed / n_images ---
    # GPU energy integrates the sampled power over the run's wall time.
    gpu_energy=$(echo "scale=2; $gpu_pavg * $run_secs" | bc -l)
    if [ -n "$n_img" ] && [ "$n_img" -gt 0 ]; then
        gpu_j_img=$(echo "scale=4; $gpu_energy / $n_img" | bc -l)
        if [ "$cpu_energy" != "NA" ]; then
            cpu_j_img=$(echo "scale=4; $cpu_energy / $n_img" | bc -l)
            tot_j_img=$(echo "scale=4; ($gpu_energy + $cpu_energy) / $n_img" | bc -l)
        else cpu_j_img="NA"; tot_j_img="NA"; fi
    else gpu_j_img="NA"; cpu_j_img="NA"; tot_j_img="NA"; fi

    printf "  FPS=%s | RSS=%s MB | GPU %sW (peak %sW) %sMiB %s%% | CPU %s%% %sW %sC | E/img GPU=%sJ CPU=%sJ tot=%sJ\n" \
        "$fps" "$peak_rss_mb" "$gpu_pavg" "$gpu_ppeak" "$gpu_mpeak" "$gpu_uavg" \
        "$cpu_util" "$cpu_pow" "$cpu_temp" "$gpu_j_img" "$cpu_j_img" "$tot_j_img"

    fps_v+=("$fps"); rss_v+=("$peak_rss_mb")
    gpow_v+=("$gpu_pavg"); gpeak_v+=("$gpu_ppeak"); gmem_v+=("$gpu_mpeak"); gutil_v+=("$gpu_uavg")
    cutil_v+=("$cpu_util")
    [ "$cpu_pow"    != "NA" ] && cpow_v+=("$cpu_pow")
    [ "$cpu_temp"   != "NA" ] && ctemp_v+=("$cpu_temp")
    [ "$gpu_j_img"  != "NA" ] && gjimg_v+=("$gpu_j_img")
    [ "$cpu_j_img"  != "NA" ] && cjimg_v+=("$cpu_j_img")
    [ "$tot_j_img"  != "NA" ] && tjimg_v+=("$tot_j_img")

    sleep 1   # let RAM/clocks settle between runs
done

# Rename the last run's outputs so backends don't overwrite each other.
[ -f "$OUTPUT_DIR/inference_results.json" ]     && mv "$OUTPUT_DIR/inference_results.json"     "$OUTPUT_DIR/inference_results_gpu_${TAG}.json"
[ -f "$OUTPUT_DIR/species_results_areas.json" ] && mv "$OUTPUT_DIR/species_results_areas.json" "$OUTPUT_DIR/species_results_gpu_${TAG}.json"

# When invoked via `sudo` (to read RAPL), hand the generated files back to the
# original user so they aren't left root-owned.
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    chown -R "$SUDO_USER" "$OUTPUT_DIR" 2>/dev/null || true
fi

# ----------------------------------------------------------------------------
# Final summary (mean +/- std)
# ----------------------------------------------------------------------------
line() { printf '%s\n' "$1"; printf '%s\n' "$1" >> "$METRICS_DIR/metrics_gpu_${TAG}.txt"; }
: > "$METRICS_DIR/metrics_gpu_${TAG}.txt"

fmt() { local a=("$@"); [ "${#a[@]}" -eq 0 ] && { echo "NA NA"; return; }; printf '%s\n' "${a[@]}" | stats; }

read -r m_fps  s_fps  < <(fmt "${fps_v[@]}")
read -r m_rss  s_rss  < <(fmt "${rss_v[@]}")
read -r m_gp   s_gp   < <(fmt "${gpow_v[@]}")
read -r m_gpk  s_gpk  < <(fmt "${gpeak_v[@]}")
read -r m_gm   s_gm   < <(fmt "${gmem_v[@]}")
read -r m_gu   s_gu   < <(fmt "${gutil_v[@]}")
read -r m_cu   s_cu   < <(fmt "${cutil_v[@]}")
read -r m_cp   s_cp   < <(fmt "${cpow_v[@]}")
read -r m_ct   s_ct   < <(fmt "${ctemp_v[@]}")
read -r m_gj   s_gj   < <(fmt "${gjimg_v[@]}")
read -r m_cj   s_cj   < <(fmt "${cjimg_v[@]}")
read -r m_tj   s_tj   < <(fmt "${tjimg_v[@]}")

line "==================================================================="
line "SUMMARY  backend=$METHOD  tag=$TAG  runs=$NUM_RUNS"
line "==================================================================="
line "Average FPS:                 $m_fps +/- $s_fps"
line "Peak process RAM (VmHWM):    $m_rss +/- $s_rss MB"
line "GPU power (avg):             $m_gp +/- $s_gp W   (peak $m_gpk W)"
line "GPU memory (peak used):      $m_gm +/- $s_gm MiB"
line "GPU utilization (avg):       $m_gu +/- $s_gu %"
line "CPU utilization (avg):       $m_cu +/- $s_cu %"
if [ "${#cpow_v[@]}" -gt 0 ]; then
    line "CPU package power (RAPL):    $m_cp +/- $s_cp W"
else
    line "CPU package power (RAPL):    N/A (run as root to read $RAPL_ENERGY)"
fi
if [ "${#ctemp_v[@]}" -gt 0 ]; then
    line "CPU temperature (avg):       $m_ct +/- $s_ct C"
else
    line "CPU temperature (avg):       N/A (sensors/Tctl not found)"
fi
line "-------------------------------------------------------------------"
line "Energy per inference (GPU):  $m_gj +/- $s_gj J/img"
if [ "${#cjimg_v[@]}" -gt 0 ]; then
    line "Energy per inference (CPU):  $m_cj +/- $s_cj J/img"
    line "Energy per inference (GPU+CPU): $m_tj +/- $s_tj J/img"
else
    line "Energy per inference (CPU):  N/A (needs root RAPL)"
fi
line "==================================================================="
line "Saved: metrics_gpu_${TAG}.txt | inference_results_gpu_${TAG}.json | species_results_gpu_${TAG}.json"
