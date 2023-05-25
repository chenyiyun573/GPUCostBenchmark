import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('power_usage.csv')

# Make timestamps start at 0
df['timestamp'] -= df['timestamp'].iloc[0]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['power_usage'])
plt.xlabel('Time (s)')
plt.ylabel('Power Usage (W)')
plt.title('Power Usage vs Time')
plt.grid(True)

# Save the figure
plt.savefig('power_usage_vs_time.png', format='png', dpi=300)
