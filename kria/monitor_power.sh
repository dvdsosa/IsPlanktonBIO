#!/bin/bash

# File to store power consumption values
TEMP_FILE="/tmp/power_values.txt"

# Remove the temporary file if it exists
if [ -f "$TEMP_FILE" ]; then
    rm "$TEMP_FILE"
fi

# Function to calculate mean value
calculate_mean() {
    local sum=0
    local count=0
    while read -r value; do
        # Ensure the value is a valid number before processing
        if [[ $value =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            sum=$(echo "$sum + $value" | bc)
            count=$((count + 1))
        fi
    done < "$TEMP_FILE"
    if [ $count -gt 0 ]; then
        mean=$(echo "scale=6; $sum / $count" | bc)
        mean=$(printf "%.2f" "$mean")
        echo "Mean power consumption: $mean W (Total counts: $count)"
    else
        echo "No valid power consumption values captured."
    fi
}

# Function to monitor power consumption
monitor_power() {
    while true; do
        cat /sys/class/hwmon/hwmon0/power1_input | awk '{print $1/1000000}' >> "$TEMP_FILE"
        sleep 0.5
    done
}

# Start monitoring in the background
monitor_power &
MONITOR_PID=$!

# Wait for a signal to stop monitoring
trap "kill $MONITOR_PID; calculate_mean; rm -f $TEMP_FILE; exit" SIGINT SIGTERM

# Keep the script running
wait
