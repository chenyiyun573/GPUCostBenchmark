import os
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pynvml import (nvmlInit,
                     nvmlDeviceGetCount,
                     nvmlDeviceGetHandleByIndex,
                     nvmlDeviceGetPowerUsage)
import threading

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

def stress_test(device, duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        # Create random tensors and perform matrix multiplication
        tensor1 = torch.rand((1000, 1000), device=device)
        tensor2 = torch.rand((1000, 1000), device=device)
        result = torch.matmul(tensor1, tensor2)

if __name__ == "__main__":
    nvmlInit()

    device_count = nvmlDeviceGetCount()

    # Start GPU stress test
    stress_threads = []
    for i in range(device_count):
        device = torch.device(f'cuda:{i}')
        torch.cuda.synchronize(device=device)
        t = threading.Thread(target=stress_test, args=(device, 600))
        t.start()
        stress_threads.append(t)

    # sample 100 times per second for 1 minute under load
    filename = "power_usage_per_ms_load.csv"
    sample_and_save(filename, device_count, 0.01, 60)
    stats_filename, plot_filename = generate_stats_and_plot(filename)
    print(f"Statistics saved to: {stats_filename}")
    print(f"Plot saved to: {plot_filename}")

    # sample once per second for 10 minutes under load
    filename = "power_usage_per_s_load.csv"
    sample_and_save(filename, device_count, 1, 600)
    stats_filename, plot_filename = generate_stats_and_plot(filename)
    print(f"Statistics saved to: {stats_filename}")
    print(f"Plot saved to: {plot_filename}")

    # Wait for all stress test threads to finish
    for t in stress_threads:
        t.join()
