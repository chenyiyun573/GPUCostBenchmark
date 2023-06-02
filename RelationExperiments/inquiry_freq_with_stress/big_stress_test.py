import time
from pynvml import *
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torch.nn as nn
import torch.multiprocessing as mp
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Initialize NVML
nvmlInit()

# Get the GPU handle
handle = nvmlDeviceGetHandleByIndex(1)

def stress_gpu():
    device = torch.device('cuda:1')
    model = models.resnet101().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    data = torch.randn(64, 3, 224, 224, device=device)
    target = torch.randn(64, 1000, device=device)
    while True:
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()

def monitor_power(sleep_interval):
    print(f"Starting power monitoring with sleep interval: {sleep_interval} seconds")
    power_usages = []
    start_time = time.time()
    while time.time() - start_time < 12:
        power = nvmlDeviceGetPowerUsage(handle)
        power_usages.append(power)
        time.sleep(sleep_interval)
    actual_freq = len(power_usages) / 12  # we divide by 60 because the sampling period is 60 seconds
    avg_power = sum(power_usages) / len(power_usages)
    print(f"Actual sampling frequency: {actual_freq} Hz")
    print(f"Average power usage at this frequency: {avg_power} mW")
    return actual_freq, avg_power

if __name__ == '__main__':
    mp.set_start_method('spawn')
    stress_process = mp.Process(target=stress_gpu)
    stress_process.start()
    sleep_intervals = [0, 0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.0011, 0.00115, 0.0012, 0.0013, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    sleep_intervals = sleep_intervals[::-1]
    freqs = []
    powers = []
    time.sleep(60)
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
