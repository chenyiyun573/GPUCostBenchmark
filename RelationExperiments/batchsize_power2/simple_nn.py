import argparse
import torch
import time
import csv
from pynvml import (nvmlInit,
                     nvmlDeviceGetHandleByIndex,
                     nvmlDeviceGetPowerUsage,
                     nvmlDeviceGetMemoryInfo,
                     nvmlDeviceGetUtilizationRates,
                     nvmlDeviceGetTemperature,
                     nvmlDeviceGetClockInfo,
                     NVML_TEMPERATURE_GPU)

nvmlClockGraphics = 0  # Hardcoded value for the graphics clock type

def parse_args():
    parser = argparse.ArgumentParser(description="A100 GPU Power Consumption Stress Test")
    parser.add_argument("--duration", type=int, default=16, help="Duration of the stress test in seconds (default: 16)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for the synthetic workload (default: 512)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the stress test on (default: cuda)")
    return parser.parse_args()

def stress_test(duration, batch_size, device):
    torch.cuda.init()
    device = torch.device(device)

    # Initialize NVML
    nvmlInit()
    device_index = 0  # Assuming we are using GPU 0
    nvml_device = nvmlDeviceGetHandleByIndex(device_index)

    model = torch.nn.Sequential(
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 4096)
    ).to(device)

    input_data = torch.randn(batch_size, 4096).to(device)
    end_time = time.time() + duration

    # Prepare lists for storing power usage, memory usage, GPU utilization, temperature, and clock frequency data along with timestamps
    power_usages = []
    memory_usages = []
    gpu_utilizations = []
    temperatures = []
    clock_frequencies = []
    timestamps = []

    while time.time() < end_time:
        output = model(input_data)
        output.backward(torch.ones_like(output))
        torch.cuda.synchronize()

        # Record power usage, memory usage, GPU utilization, temperature, clock frequency, and timestamp during the iteration
        power_usage = nvmlDeviceGetPowerUsage(nvml_device) / 1000.0  # Convert mW to W
        power_usages.append(power_usage)

        memory_info = nvmlDeviceGetMemoryInfo(nvml_device)
        memory_usage = memory_info.used / 1024**2  # Convert to MB
        memory_usages.append(memory_usage)

        gpu_utilization = nvmlDeviceGetUtilizationRates(nvml_device).gpu
        gpu_utilizations.append(gpu_utilization)

        temperature = nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU)
        temperatures.append(temperature)

        clock_frequency = nvmlDeviceGetClockInfo(nvml_device, nvmlClockGraphics)
        clock_frequencies.append(clock_frequency)

        timestamp = time.time()
        timestamps.append(timestamp)

    # Write power usage, memory usage, GPU utilization, temperature, clock frequency data along with timestamps to a CSV file
    with open('gpu_usage.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'power_usage', 'memory_usage', 'gpu_utilization', 'temperature', 'clock_frequency'])
        for t, p, m, g, te, c in zip(timestamps, power_usages, memory_usages, gpu_utilizations, temperatures, clock_frequencies):
            writer.writerow([t, p, m, g, te, c])

    import matplotlib.pyplot as plt

    # Convert timestamps to elapsed time
    start_time = timestamps[0]
    elapsed_time = [t - start_time for t in timestamps]

    # Create a plot of power usage, memory usage, GPU utilization, temperature, and clock frequency over time
    plt.figure()
    plt.plot(elapsed_time, power_usages, label='Power Usage')
    plt.plot(elapsed_time, memory_usages, label='Memory Usage')
    plt.plot(elapsed_time, gpu_utilizations, label='GPU Utilization')
    plt.plot(elapsed_time, temperatures, label='Temperature')
    plt.plot(elapsed_time, clock_frequencies, label='Clock Frequency')
    plt.title('GPU Usage Over Time')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Usage')
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG image
    plt.savefig('gpu_usage.png')


if __name__ == "__main__":
    args = parse_args()
    stress_test(args.duration, args.batch_size, args.device)
