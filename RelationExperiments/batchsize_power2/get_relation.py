import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import time

# Define the batch sizes you want to test
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 16384*2, 16384*4, 16384*8, 16384*16]

# Create results directory if it doesn't exist
os.makedirs('./results', exist_ok=True)

# Prepare the CSV files for storing statistical summaries
statistics = ['power_usage', 'memory_usage', 'gpu_utilization', 'temperature', 'clock_frequency']
for stat in statistics:
    with open(f'./results/{stat}_statistics.csv', 'w') as file:
        file.write('batch_size,mean,std,min,25%,50%,75%,max\n')

for batch_size in batch_sizes:
    # Create a directory to store the results for this batch size
    directory = f'./plot/batch_{batch_size}'
    os.makedirs(directory, exist_ok=True)

    # Run the script with the given batch size
    subprocess.run(['python3', 'simple_nn.py', '--duration', '16', '--batch-size', str(batch_size), '--device', 'cuda:0'])

    # Check if the files exist in the destination directory
    while not (os.path.exists('gpu_usage.csv') and os.path.exists('gpu_usage.png')):
        time.sleep(1)

    # Move the csv and png files to the directory
    os.rename('gpu_usage.csv', f'{directory}/gpu_usage.csv')
    os.rename('gpu_usage.png', f'{directory}/gpu_usage.png')

    # Calculate statistical summary of power usage, memory usage, GPU utilization, temperature, clock frequency and write to the summary csv file
    gpu_usage_data = pd.read_csv(f'{directory}/gpu_usage.csv')
    for stat in statistics:
        usage_summary = gpu_usage_data[stat].describe()
        with open(f'./results/{stat}_statistics.csv', 'a') as file:
            file.write(f'{batch_size},{usage_summary["mean"]},{usage_summary["std"]},{usage_summary["min"]},{usage_summary["25%"]},{usage_summary["50%"]},{usage_summary["75%"]},{usage_summary["max"]}\n')

    print(f'Experiment done for batch size {batch_size}')

# Read the statistical summary files and plot mean usage against batch size
for stat in statistics:
    stats = pd.read_csv(f'./results/{stat}_statistics.csv')
    plt.figure()
    plt.plot(stats['batch_size'], stats['mean'], marker='o')
    plt.xscale('log', basex=2)
    plt.title(f'Mean {stat.replace("_", " ").capitalize()} vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel(f'Mean {stat.replace("_", " ").capitalize()}')
    plt.grid(True)
    plt.savefig(f'./results/mean_{stat}_usage.png')

