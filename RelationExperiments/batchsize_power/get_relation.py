import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import time

# Define the batch sizes you want to test
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Create results directory if it doesn't exist
os.makedirs('./results', exist_ok=True)

# Prepare the CSV file for storing statistical summaries
with open('./results/power_statistics.csv', 'w') as file:
    file.write('batch_size,mean,std,min,25%,50%,75%,max\n')

for batch_size in batch_sizes:
    # Create a directory to store the results for this batch size
    directory = f'./plot/batch_{batch_size}'
    os.makedirs(directory, exist_ok=True)

    # Run the script with the given batch size
    subprocess.run(['python3', 'simple_nn.py', '--duration', '16', '--batch-size', str(batch_size), '--device', 'cuda:0'])

    # Check if the files exist in the destination directory
    while not (os.path.exists('power_usage.csv') and os.path.exists('power_usage.png')):
        time.sleep(1)  # Wait for 1 second before checking again

    # Move the csv and png files to the directory
    os.rename('power_usage.csv', f'{directory}/power_usage.csv')
    os.rename('power_usage.png', f'{directory}/power_usage.png')

    # Calculate statistical summary of power usage and write to the summary csv file
    power_data = pd.read_csv(f'{directory}/power_usage.csv')
    power_summary = power_data['power_usage'].describe()
    with open('./results/power_statistics.csv', 'a') as file:
        file.write(f'{batch_size},{power_summary["mean"]},{power_summary["std"]},{power_summary["min"]},{power_summary["25%"]},{power_summary["50%"]},{power_summary["75%"]},{power_summary["max"]}\n')

    print(f'One experimet done for batch size {batch_size}')

# Read the statistical summary file
stats = pd.read_csv('./results/power_statistics.csv')

# Plot mean power usage against batch size
plt.figure()
plt.plot(stats['batch_size'], stats['mean'], marker='o')
plt.xscale('log', basex=2)  # Use log scale for batch size
plt.title('Mean Power Usage vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Mean Power Usage (W)')
plt.grid(True)
plt.savefig('./results/mean_power_usage.png')

# Plot standard deviation of power usage against batch size
plt.figure()
plt.plot(stats['batch_size'], stats['std'], marker='o')
plt.xscale('log', basex=2)  # Use log scale for batch size
plt.title('Standard Deviation of Power Usage vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Standard Deviation of Power Usage (W)')
plt.grid(True)
plt.savefig('./results/std_power_usage.png')
