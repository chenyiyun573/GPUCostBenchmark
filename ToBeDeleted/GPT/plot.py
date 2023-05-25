import pandas as pd
import matplotlib.pyplot as plt

# Load data from csv, skip the header row
data = pd.read_csv('power_usage.csv', skiprows=1, names=['timestamp', 'power_usage'])

# Convert string to float and subtract the start time
start_time = float(data['timestamp'].min())
data['timestamp'] = data['timestamp'].apply(lambda x: (float(x) - start_time))

# Plot data
plt.figure(figsize=(10, 5))
plt.plot(data['timestamp'], data['power_usage'])

plt.title('Power usage vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Power usage')
plt.grid()

# Save the figure to a file
plt.savefig('power_usage_plot.png')
