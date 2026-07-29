#!/bin/bash
#
# Mide metricas del sistema en estado IDLE (solo Linux, sin aplicacion de inference):
#   - Memoria        (VmRSS promedio durante el idle)
#   - Potencia       (W medios, leidos de /sys/class/hwmon/hwmon0/power1_input)
#   - Energia        (no aplica, se reporta 0)
#
# La tabla de metricas se guarda en:  output/metrics/metrics_paper2_idle.txt
#
# Uso:   ./run_paper2_idle_experiments.sh [idle_duration_seconds]
#        (idle_duration_seconds por defecto = 120)
#
set -uo pipefail

BASE_DIR=/home/petalinux/vai_runtime
OUTPUT_DIR=$BASE_DIR/output
METRICS_DIR=$OUTPUT_DIR/metrics
METRICS_FILE=$METRICS_DIR/metrics_paper2_idle.txt

# Duracion del idle (argumento opcional, por defecto 120 segundos)
idle_duration=${1:-120}

cd "$BASE_DIR"
mkdir -p "$METRICS_DIR"

# Todo lo que imprima este script se muestra en pantalla y se guarda en el .txt
exec > >(tee "$METRICS_FILE") 2>&1

echo "==========================================================="
echo "paper2 IDLE-OS experiment  -  $(date)"
echo "Idle duration: $idle_duration seconds"
echo "==========================================================="

# ── Arranca el monitor de potencia ──
POWER_MON_OUT=/tmp/power_monitor_paper2_idle.out
./monitor_power.sh > "$POWER_MON_OUT" 2>&1 &
MONITOR_PID=$!

# Acumuladores para memoria
sum_mem=0
mem_values=()
sample_count=0

echo "-----------------------------------"
echo "Starting $idle_duration second idle measurement..."
echo "-----------------------------------"

# Muestrear memoria cada 0.5 segundos durante idle_duration
start_time=$(date +%s)
while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    if [ $elapsed -ge $idle_duration ]; then
        break
    fi

    # Obtener uso de memoria del sistema (MemAvailable)
    mem_available=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')

    if [ -n "$mem_total" ] && [ -n "$mem_available" ]; then
        mem_used_kb=$((mem_total - mem_available))
        mem_used_mb=$(echo "scale=2; $mem_used_kb / 1024" | bc)

        mem_values+=("$mem_used_mb")
        sum_mem=$(echo "$sum_mem + $mem_used_mb" | bc)
        sample_count=$((sample_count + 1))
    fi

    sleep 0.5
done

echo "[OK] Idle measurement finished"

# ── Paramos el monitor de potencia y leemos su media ────────────────────────
kill -TERM $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null
mean_power=$(grep "Mean power consumption" "$POWER_MON_OUT" | awk '{print $4}')
[ -z "$mean_power" ] && mean_power=0

# ── Media y desviacion tipica de memoria ──────────────────────────────────────
mean_mem=0
std_mem=0
if [ $sample_count -gt 0 ]; then
    mean_mem=$(echo "scale=2; $sum_mem / $sample_count" | bc)

    if [ $sample_count -gt 1 ]; then
        sq_mem=0
        for v in "${mem_values[@]}"; do
            sq_mem=$(echo "$sq_mem + ($v - $mean_mem)^2" | bc -l)
        done
        std_mem=$(echo "scale=2; sqrt($sq_mem / ($sample_count - 1))" | bc -l)
    fi
fi

echo "==================================="
echo "RESUMEN FINAL paper2 IDLE-OS"
echo "==================================="
echo "Idle duration:               $idle_duration seconds"
echo "Memory used (avg):           $mean_mem +/- $std_mem MB (std)"
echo "Memory samples:              $sample_count"
echo "Mean power consumption:      $mean_power W"
echo "Energy per image:            0 J/img (N/A for idle)"
echo "==================================="
echo "Metricas guardadas en: $METRICS_FILE"
echo "==========================================================="
