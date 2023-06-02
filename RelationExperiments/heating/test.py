import time
from pynvml import *
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import os

# Initialize NVML
nvmlInit()

# Get the GPU handle
handle = nvmlDeviceGetHandleByIndex(1)

def stress_gpu(duration):
    device = torch.device('cuda:1')
    end_time = time.time() + duration
    while time.time() < end_time:
        # Create random tensors and perform matrix multiplication
        tensor1 = torch.rand((1000, 1000), device=device)
        tensor2 = torch.rand((1000, 1000), device=device)
        result = torch.matmul(tensor1, tensor2)

def monitor_power(sleep_interval, duration):
    print(f"Starting power monitoring with sleep interval: {sleep_interval} seconds")
    power_usages = []
    timestamps = []
    end_time = time.time() + duration
    while time.time() < end_time:
        power = nvmlDeviceGetPowerUsage(handle)
        power_usages.append(power)
        timestamps.append(time.time())
        time.sleep(sleep_interval)
    return timestamps, power_usages

if __name__ == '__main__':
    mp.set_start_method('spawn')
    duration = 120  # duration in seconds
    sleep_interval = 0.005  # sleep interval in seconds
    stress_process = mp.Process(target=stress_gpu, args=(duration,))
    stress_process.start()
    #time.sleep(10)
    timestamps, powers = monitor_power(sleep_interval, duration)
    stress_process.terminate()
    plt.figure()
    plt.plot(timestamps, powers)
    plt.xlabel('Time (s)')
    plt.ylabel('Power Usage (mW)')
    plt.title('Power Usage vs Time')
    os.makedirs('results', exist_ok=True)
    plt.savefig('./results/power_vs_time.png')
    plt.close()
    nvmlShutdown()
