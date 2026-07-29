#!/bin/bash
#
# Ejecuta el experimento de paper2 en la Kria y extrae metricas de:
#   - FPS            (media +/- std sobre N runs)
#   - Memoria VmHWM  (media +/- std sobre N runs)
#   - Potencia       (W medios, leidos de /sys/class/hwmon/hwmon0/power1_input)
#   - Energia        (J/imagen = potencia_media / fps_media)
#
# Mismo proceso que run_all_kria_le_experiments.sh + resnet50-power-memory.sh
# + monitor_power.sh, pero ejecutando UN solo binario:
#   ./bin/resnet50_cosine_faiss_turbojpeg_IVFPQ_fix_paper2 --config configs/default_config_paper2.yaml
#
# La tabla de metricas se guarda en:  output/metrics/metrics_paper2.txt
# (mismo lugar que en run_all_kria_le_experiments.sh)
#
# Uso:   ./run_paper2_experiments.sh [num_runs]     (num_runs por defecto = 10)
#
set -uo pipefail

BASE_DIR=/home/petalinux/vai_runtime
OUTPUT_DIR=$BASE_DIR/output
METRICS_DIR=$OUTPUT_DIR/metrics
METRICS_FILE=$METRICS_DIR/metrics_paper2.txt

CMD="./bin/resnet50_cosine_faiss_turbojpeg_IVFPQ_fix_paper2 --config configs/default_config_paper2.yaml"

# Numero de runs (argumento opcional, por defecto 10)
num_runs=${1:-10}

cd "$BASE_DIR"
mkdir -p "$METRICS_DIR"

# Todo lo que imprima este script se muestra en pantalla y se guarda en el .txt
exec > >(tee "$METRICS_FILE") 2>&1

echo "==========================================================="
echo "paper2 experiment  -  $(date)"
echo "Comando: $CMD"
echo "Runs:    $num_runs"
echo "==========================================================="

# ── Arranca el monitor de potencia (captura su salida para calcular joules) ──
POWER_MON_OUT=/tmp/power_monitor_paper2.out
./monitor_power.sh > "$POWER_MON_OUT" 2>&1 &
MONITOR_PID=$!

# Acumuladores
sum_fps=0
sum_mem=0
fps_values=()
mem_values=()

for ((i=1; i<=num_runs; i++)); do
    echo "-----------------------------------"
    echo "Run #$i"

    # Lanzamos el binario en segundo plano y monitorizamos su pico de RAM (VmHWM)
    $CMD > run_output_paper2.tmp 2>&1 &
    CMD_PID=$!

    peak_mem_kb=0
    while kill -0 $CMD_PID 2>/dev/null; do
        current_hwm=$(grep VmHWM /proc/$CMD_PID/status 2>/dev/null | awk '{print $2}')
        if [ -n "$current_hwm" ] && [ "$current_hwm" -gt "$peak_mem_kb" ]; then
            peak_mem_kb=$current_hwm
        fi
        sleep 0.2
    done
    wait $CMD_PID

    # FPS desde la salida del binario ("Average FPS: <valor>")
    fps=$(grep "Average FPS:" run_output_paper2.tmp | awk '{print $3}')
    if [ -z "$fps" ]; then fps=0; echo "Aviso: No se encontraron FPS en Run #$i"; fi
    if [ -z "$peak_mem_kb" ]; then peak_mem_kb=0; fi

    peak_mem_mb=$(echo "scale=2; $peak_mem_kb / 1024" | bc)

    echo "Run #$i finished | Avg FPS = $fps | Peak Memory (VmHWM) = $peak_mem_mb MB"

    fps_values+=("$fps")
    mem_values+=("$peak_mem_mb")
    sum_fps=$(echo "$sum_fps + $fps" | bc)
    sum_mem=$(echo "$sum_mem + $peak_mem_mb" | bc)

    sleep 1
done

# ── Paramos el monitor de potencia y leemos su media ────────────────────────
kill -TERM $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null
mean_power=$(grep "Mean power consumption" "$POWER_MON_OUT" | awk '{print $4}')
[ -z "$mean_power" ] && mean_power=0

rm -f run_output_paper2.tmp

# ── Medias y desviaciones tipicas (divisor N-1) ─────────────────────────────
mean_fps=$(echo "scale=4; $sum_fps / $num_runs" | bc)
mean_mem=$(echo "scale=2; $sum_mem / $num_runs" | bc)

std_fps=0
std_mem=0
if [ "$num_runs" -gt 1 ]; then
    sq_fps=0
    for v in "${fps_values[@]}"; do
        sq_fps=$(echo "$sq_fps + ($v - $mean_fps)^2" | bc -l)
    done
    std_fps=$(echo "scale=4; sqrt($sq_fps / ($num_runs - 1))" | bc -l)

    sq_mem=0
    for v in "${mem_values[@]}"; do
        sq_mem=$(echo "$sq_mem + ($v - $mean_mem)^2" | bc -l)
    done
    std_mem=$(echo "scale=2; sqrt($sq_mem / ($num_runs - 1))" | bc -l)
fi

# ── Energia por imagen (J/img) = Potencia media (W) / FPS medios ────────────
joules_per_img=0
if [ "$(echo "$mean_fps > 0" | bc -l)" -eq 1 ]; then
    joules_per_img=$(echo "scale=4; $mean_power / $mean_fps" | bc -l)
fi

echo "==================================="
echo "RESUMEN FINAL paper2 ($num_runs runs)"
echo "==================================="
echo "Average FPS:              $mean_fps +/- $std_fps (std)"
echo "Memory Footprint (VmHWM): $mean_mem +/- $std_mem MB (std)"
echo "Mean power consumption:   $mean_power W"
echo "Energy per image:         $joules_per_img J/img"
echo "==================================="

# ── Renombrar trace.json (el binario lo escribe en BASE_DIR/trace.json) ─────
if [ -f "$BASE_DIR/trace.json" ]; then
    mv "$BASE_DIR/trace.json" "$OUTPUT_DIR/trace_paper2.json"
    echo "[OK] trace.json -> output/trace_paper2.json"
else
    echo "[WARN] trace.json no encontrado"
fi

# Los JSON de resultados ya se escriben con nombre paper2 desde el YAML:
#   output/inference_results_paper2.json   (species_json)
#   output/species_results_areas_paper2.json (areas_json)
echo "Metricas guardadas en: $METRICS_FILE"
echo "==========================================================="
