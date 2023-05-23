"""
README here: 

run this script with:
python3 pytorch.py --duration 300 --batch-size 512 --device cuda:2

Time: 20230522 16:12
Author: v-yiyunchen@microsoft.com
"""

import os
import argparse
import torch
import time
import csv
from pynvml import (nvmlInit,
                     nvmlDeviceGetHandleByIndex,
                     nvmlDeviceGetPowerUsage,
                     nvmlDeviceGetMemoryInfo)

def parse_args():
    parser = argparse.ArgumentParser(description="A100 GPU Power Consumption Stress Test")
    parser.add_argument("--duration", type=int, default=16, help="Duration of the stress test in seconds (default: 16)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for the synthetic workload (default: 512)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the stress test on (default: cuda)")
    return parser.parse_args()

def stress_test(duration, batch_size, device):
    os.mkdir('in')

    torch.cuda.init()
    device = torch.device(device)
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
    ).to(device)

    input_data = torch.randn(batch_size, 4096).to(device)
    end_time = time.time() + duration

    # Prepare lists for storing power usage data and timestamps
    power_usages = []
    memory_usages = []
    timestamps = []

    while time.time() < end_time:
        output = model(input_data)
        output.backward(torch.ones_like(output))
        torch.cuda.synchronize()

        # Record power usage, memory usage, and timestamp during the iteration
        power_usage = nvmlDeviceGetPowerUsage(nvml_device) / 1000.0  # Convert mW to W
        power_usages.append(power_usage)

        memory_info = nvmlDeviceGetMemoryInfo(nvml_device)
        memory_usage = memory_info.used / 1024**2  # Convert to MB
        memory_usages.append(memory_usage)

        timestamp = time.time()
        timestamps.append(timestamp)

    # Write power usage data and timestamps to a CSV file
    with open('./in/power_usage.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'power_usage'])
        for t, p in zip(timestamps, power_usages):
            writer.writerow([t, p])

    import matplotlib.pyplot as plt

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
    plt.savefig('./in/power_usage.png')


if __name__ == "__main__":
    args = parse_args()
    stress_test(args.duration, args.batch_size, args.device)
