import pandas as pd
import matplotlib.pyplot as plt

# Read the data from a CSV file.
data = pd.read_csv('powerData.csv')

# Create a new figure and set the size of the figure.
fig, ax = plt.subplots(figsize=(10, 6))

# Plot power versus time.
ax.plot(data['Time(ms)'], data['Power(mW)'])

# Set the labels for the x-axis and y-axis.
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Power (mW)')

# Set the title of the plot.
ax.set_title('Power vs Time')

# Save the plot to a file. The format is determined by the extension of the file name.
plt.savefig('power_vs_time.png', dpi=300, bbox_inches='tight')
