import torch
from transformers import GPT2LMHeadModel
import time
import csv
from pynvml import (nvmlInit,
                     nvmlDeviceGetHandleByIndex,
                     nvmlDeviceGetPowerUsage,
                     nvmlDeviceGetMemoryInfo)

# Specify the GPT-2 model and other parameters
model_type = 'gpt2'
input_length = 64  # This corresponds to the number of tokens in the sequence
batch_size = 32
num_iterations = 5000
num_warmup_iterations = 100

# Initialize NVML
nvmlInit()
device_index = 0  # Assuming we are using GPU 0
nvml_device = nvmlDeviceGetHandleByIndex(device_index)

# Create the model and move it to GPU if available
model = GPT2LMHeadModel.from_pretrained(model_type)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create random input tensor for benchmarking
input_shape = (batch_size, input_length)
input_tensor = torch.randint(0, model.config.vocab_size, input_shape, device=device)

# Set the model to evaluation mode
model.eval()

# Warm-up run to ensure GPU is fully initialized
with torch.no_grad():
    for _ in range(num_warmup_iterations):
        output = model(input_tensor)
    torch.cuda.synchronize()  # Ensure all operations are completed

# Start the benchmark
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()

# Start power and memory monitoring
power_usages = []
memory_usages = []
timestamps = []

for _ in range(num_iterations):
    with torch.no_grad():
        # Forward pass
        output = model(input_tensor)
        
    # Record power usage, memory usage, and timestamp during the iteration
    power_usage = nvmlDeviceGetPowerUsage(nvml_device) / 1000.0  # Convert mW to W
    power_usages.append(power_usage)
    
    memory_info = nvmlDeviceGetMemoryInfo(nvml_device)
    memory_usage = memory_info.used / 1024**2  # Convert to MB
    memory_usages.append(memory_usage)

    timestamp = time.time()
    timestamps.append(timestamp)

end_event.record()
torch.cuda.synchronize()  # Ensure all operations are completed

# Write power usage data and timestamps to a CSV file
with open('power_usage.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'power_usage'])
    for t, p in zip(timestamps, power_usages):
        writer.writerow([t, p])

# Measure time using CUDA events
total_time_cuda = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds

# Calculate the average time, power usage, memory usage, and FPS
average_time_cuda = total_time_cuda / num_iterations
average_power_usage = sum(power_usages) / len(power_usages)
average_memory_usage = sum(memory_usages) / len(memory_usages)
fps_cuda = batch_size / average_time_cuda

# Print benchmark results
print(f'Model: {model_type}')
print(f'Input Length: {input_length} tokens')
print(f'Batch Size: {batch_size}')
print(f'Number of Iterations: {num_iterations}')
print(f'Total GPU Time: {total_time_cuda:.4f} seconds')
print(f'Average GPU Time per Iteration: {average_time_cuda:.4f} seconds')
print(f'FPS: {fps_cuda:.2f}')
print(f'Average Power Usage: {average_power_usage:.2f} W')
print(f'Average Memory Usage: {average_memory_usage:.2f} MB')
