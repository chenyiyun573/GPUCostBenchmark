"""
    This script is aimed to measure the idle stage power consumption of GPUs.
    run it with:

    nohup python3 idle_measure.py &

    and wait for 10 minutes to get the idle stage power data.
    
    The default sampling frequecy: 100Hz for 1 minutes and 1Hz for 10 minutes.

"""
import os
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pynvml import (nvmlInit,
                     nvmlDeviceGetCount,
                     nvmlDeviceGetHandleByIndex,
                     nvmlDeviceGetPowerUsage)

# Define results directory
results_dir = './results/'

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

def get_power_usage(device_count):
    power_usages = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        power_usage = nvmlDeviceGetPowerUsage(handle)
        power_usages.append(power_usage / 1000.0)  # milliwatts to watts
    return power_usages

def sample_and_save(filename, device_count, interval, duration):
    # Add the results directory to the filename
    filename = os.path.join(results_dir, filename)
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time"] + ["GPU"+str(i) for i in range(device_count)])

        start_time = time.perf_counter()
        while time.perf_counter() - start_time < duration:
            current_time = time.perf_counter()
            writer.writerow([current_time - start_time] + get_power_usage(device_count))
            while time.perf_counter() - current_time < interval:
                pass

def generate_stats_and_plot(filename):
    # Add the results directory to the filename
    filename = os.path.join(results_dir, filename)
    
    df = pd.read_csv(filename, index_col='time')

    stats_filename = filename.replace('.csv', '_stats.csv')
    df_stats = df.describe()
    df_stats.to_csv(stats_filename)
    
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)

    plt.title('GPU Power Usage Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Power Usage (W)')
    plt.legend()
    plt.grid(True)
    
    plot_filename = filename.replace('.csv', '.png')
    plt.savefig(plot_filename)
    
    return stats_filename, plot_filename

if __name__ == "__main__":
    nvmlInit()

    device_count = nvmlDeviceGetCount()

    # sample 100 times per second for 1 minute
    filename = "power_usage_per_ms.csv"
    sample_and_save(filename, device_count, 0.01, 60)
    stats_filename, plot_filename = generate_stats_and_plot(filename)
    print(f"Statistics saved to: {stats_filename}")
    print(f"Plot saved to: {plot_filename}")

    # sample once per second for 10 minutes
    filename = "power_usage_per_s.csv"
    sample_and_save(filename, device_count, 1, 600)
    stats_filename, plot_filename = generate_stats_and_plot(filename)
    print(f"Statistics saved to: {stats_filename}")
    print(f"Plot saved to: {plot_filename}")
