import os
import argparse
import torch
import time
import csv
import matplotlib.pyplot as plt
from pynvml import (nvmlInit,
                     nvmlDeviceGetHandleByIndex,
                     nvmlDeviceGetPowerUsage,
                     nvmlDeviceGetMemoryInfo)
import multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(description="A100 GPU Power Consumption Stress Test")
    parser.add_argument("--duration", type=int, default=16, help="Duration of the stress test in seconds (default: 16)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for the synthetic workload (default: 512)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the stress test on (default: cuda)")
    return parser.parse_args()

def get_power_data(device_index, power_usage, memory_usage, event):
    nvmlInit()
    nvml_device = nvmlDeviceGetHandleByIndex(device_index)
    while not event.is_set():
        power_usage.value = nvmlDeviceGetPowerUsage(nvml_device)  # Power usage in milliwatts
        memory_info = nvmlDeviceGetMemoryInfo(nvml_device)
        memory_usage.value = memory_info.used
        time.sleep(0.1)

def stress_test(duration, batch_size, device):
    if not os.path.exists('out'):
        os.mkdir('out')

    torch.cuda.init()
    device_name = torch.device(device)
    gpu_count = torch.cuda.device_count()

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
    ).to(device_name)

    input_data = torch.randn(batch_size, 4096).to(device_name)
    end_time = time.time() + duration

    # Create shared variables for power and memory data
    power_usage = mp.Value('f', 0.0)
    memory_usage = mp.Value('L', 0)

    # Create an event to signal when to stop collecting power data
    event = mp.Event()

    # Start the process for power data inquiry
    power_process = mp.Process(target=get_power_data, args=(device_index, power_usage, memory_usage, event))
    power_process.start()

    # Prepare lists for storing power usage data and timestamps
    power_usages = []
    memory_usages = []
    timestamps = []

    while time.time() < end_time:
        output = model(input_data)
        output.backward(torch.ones_like(output))
        torch.cuda.synchronize()

        # Record timestamp during the iteration
        timestamp = time.time()
        timestamps.append(timestamp)

        # Retrieve power and memory data
        power_usages.append(power_usage.value / 1000.0)  # Convert to watts
        memory_usages.append(memory_usage.value / (1024 ** 2))  # Convert to megabytes

    # Set event to signal the power process to terminate
    event.set()

    # Calculate sampling frequency
    sampling_frequency = len(timestamps) / duration
    print("Sampling Frequency:", sampling_frequency, "Hz")

    # Write power usage data and timestamps to a CSV file
    with open('./out/power_usage.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'power_usage'])
        for t, p in zip(timestamps, power_usages):
            writer.writerow([t, p])

    # Convert timestamps to elapsed time
    start_time = timestamps[0]
    elapsed_time = [t - start_time for t in timestamps]

    # Create a plot of power usage over time
    plt.figure()
    plt.plot(elapsed_time, power_usages)
    plt.title('Power Usage Over Time')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Power Usage (W)')
    plt.grid(True)

    # Save the plot as a PNG image
    plt.savefig('./out/power_usage.png')

    # Join the power process
    power_process.join()


if __name__ == "__main__":
    args = parse_args()
    stress_test(args.duration, args.batch_size, args.device)
