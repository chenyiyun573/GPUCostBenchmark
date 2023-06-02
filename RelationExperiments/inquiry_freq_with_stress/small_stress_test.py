import time
from pynvml import *
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def monitor_power(sleep_interval):
    print(f"Starting power monitoring with sleep interval: {sleep_interval} seconds")
    power_usages = []
    start_time = time.time()
    while time.time() - start_time < 5:
        power = nvmlDeviceGetPowerUsage(handle)
        power_usages.append(power)
        time.sleep(sleep_interval)
    actual_freq = len(power_usages) / 5  # we divide by 5 because the sampling period is 5 seconds
    avg_power = sum(power_usages) / len(power_usages)
    print(f"Actual sampling frequency: {actual_freq} Hz")
    print(f"Average power usage at this frequency: {avg_power} mW")
    return actual_freq, avg_power

if __name__ == '__main__':
    mp.set_start_method('spawn')
    duration = 1800  # duration in seconds
    stress_process = mp.Process(target=stress_gpu, args=(duration,))
    stress_process.start()
    sleep_intervals = [0, 0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.0011, 0.00115, 0.0012, 0.0013, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    #sleep_intervals = sleep_intervals[::-1]
    freqs = []
    powers = []
    time.sleep(10)
    for interval in sleep_intervals:
        freq, power = monitor_power(interval)
        freqs.append(freq)
        powers.append(power)
    stress_process.terminate()
    plt.figure()
    plt.plot(freqs, powers, marker='o')
    plt.xlabel('Sampling Frequency (Hz)')
    plt.ylabel('Average Power Usage (mW)')
    plt.title('Average Power Usage vs Sampling Frequency')
    os.makedirs('results', exist_ok=True)
    plt.savefig('./results/power_vs_frequency.png')
    plt.close()
    nvmlShutdown()
