import os
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pynvml import (nvmlInit,
                     nvmlDeviceGetCount,
                     nvmlDeviceGetHandleByIndex,
                     nvmlDeviceGetPowerUsage,
                     nvmlDeviceGetTemperature,
                     nvmlDeviceGetUtilizationRates,
                     nvmlDeviceGetMemoryInfo,
                     nvmlDeviceGetClockInfo,
                     NVML_CLOCK_GRAPHICS)

# Define results directory
results_dir = './results/'

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

def get_device_data(device_count):
    device_data = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        power_usage = nvmlDeviceGetPowerUsage(handle) / 1000.0  # milliwatts to watts
        temperature = nvmlDeviceGetTemperature(handle, 0)  # Temperature in Celsius
        utilization = nvmlDeviceGetUtilizationRates(handle).gpu  # GPU utilization in percent
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        memory = memory_info.used / memory_info.total * 100  # Memory usage in percent
        clock_freq = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)  # Clock frequency in MHz
        device_data.append((power_usage, temperature, utilization, memory, clock_freq))
    return device_data

def sample_and_save(filename, device_count, interval, duration):
    # Add the results directory to the filename
    filename = os.path.join(results_dir, filename)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['time']
        for i in range(device_count):
            header.extend([f'GPU{i}_power', f'GPU{i}_temp', f'GPU{i}_util', f'GPU{i}_mem', f'GPU{i}_clock'])
        writer.writerow(header)

        start_time = time.perf_counter()
        while time.perf_counter() - start_time < duration:
            current_time = time.perf_counter()
            row = [current_time - start_time]
            for data in get_device_data(device_count):
                row.extend(data)
            writer.writerow(row)
            while time.perf_counter() - current_time < interval:
                pass

def generate_stats_and_plot(filename):
    # Add the results directory to the filename
    filename = os.path.join(results_dir, filename)

    df = pd.read_csv(filename, index_col='time')

    stats_filename = filename.replace('.csv', '_stats.csv')
    df_stats = df.describe()
    df_stats.to_csv(stats_filename)

    fig, axs = plt.subplots(5, figsize=(10, 30))
    cols = df.columns
    for i, ax in enumerate(axs):
        for j in range(device_count):
            ax.plot(df.index, df[cols[i + 5 * j]], label=cols[i + 5 * j])
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)
    axs[0].set_ylabel('Power Usage (W)')
    axs[1].set_ylabel('Temperature (C)')
    axs[2].set_ylabel('Utilization (%)')
    axs[3].set_ylabel('Memory Usage (%)')
    axs[4].set_ylabel('Clock Frequency (MHz)')

    fig.tight_layout()
    plot_filename = filename.replace('.csv', '.png')
    plt.savefig(plot_filename)

    return stats_filename, plot_filename

if __name__ == "__main__":
    nvmlInit()

    device_count = nvmlDeviceGetCount()

    # sample 100 times per second for 1 minute
    filename = "device_data_per_ms.csv"
    sample_and_save(filename, device_count, 0.01, 60)
    stats_filename, plot_filename = generate_stats_and_plot(filename)
    print(f"Statistics saved to: {stats_filename}")
    print(f"Plot saved to: {plot_filename}")

    # sample once per second for 10 minutes
    filename = "device_data_per_s.csv"
    sample_and_save(filename, device_count, 1, 600)
    stats_filename, plot_filename = generate_stats_and_plot(filename)
    print(f"Statistics saved to: {stats_filename}")
    print(f"Plot saved to: {plot_filename}")
