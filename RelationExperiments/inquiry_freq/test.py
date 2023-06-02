import time
from pynvml import *
import matplotlib.pyplot as plt

# Initialize NVML
nvmlInit()

# Get the first GPU handle. If you have multiple GPUs, you may need to loop over this
handle = nvmlDeviceGetHandleByIndex(7)

# Define the desired sleep intervals in seconds (e.g., 1.0, 0.1, 0.01, 0.001 for 1Hz, 10Hz, 100Hz, 1kHz desired rates)
sleep_intervals = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.003, 0.002, 0.0015, 0.0013, 0.0012, 0.00115, 0.0011, 0.001, 0.0008, 0.0005, 0.0003, 0.0001, 0]
sleep_intervals = sleep_intervals[::-1]
freqs = []
powers = []

# Loop over the different sleep intervals
for interval in sleep_intervals:
    print(f"Attempting sampling with sleep interval: {interval} seconds")

    # Initialize a list to store the power usage samples
    power_usages = []

    # Start the timer
    start_time = time.time()

    # Run for a specified duration, here we use 5 seconds as per your request
    while time.time() - start_time < 5:
        # Get the current power usage in milliwatts
        power = nvmlDeviceGetPowerUsage(handle)

        # Append the power usage to the list
        power_usages.append(power)

        # Sleep for the appropriate amount of time
        time.sleep(interval)

    # Calculate the actual sampling frequency based on number of samples taken during the period
    actual_freq = len(power_usages) / 5  # we divide by 5 because the sampling period is 5 seconds

    # Print the actual sampling frequency and the average power usage over the sampling period
    avg_power = sum(power_usages) / len(power_usages)
    print(f"Actual sampling frequency: {actual_freq} Hz")
    freqs.append(actual_freq)
    print(f"Average power usage at this frequency: {avg_power} mW")
    powers.append(avg_power)


# Shut down NVML
nvmlShutdown()


# Create the plot
plt.figure()
plt.plot(freqs, powers, marker='o')

# Add labels and title
plt.xlabel('Sampling Frequency (Hz)')
plt.ylabel('Average Power Usage (mW)')
plt.title('Average Power Usage vs Sampling Frequency')

# Save the plot as a PNG file
os.makedirs('results', exist_ok=True)
plt.savefig('./results/power_vs_frequency.png')

plt.close()