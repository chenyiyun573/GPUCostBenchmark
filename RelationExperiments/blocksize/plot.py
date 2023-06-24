import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory where the CSV files are located
data_dir = "results"
plot_dir = "results_plot"

# Create the plot directory if it does not exist
os.makedirs(plot_dir, exist_ok=True)

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Process each file
for csv_file in csv_files:
    # Load the data from the CSV file
    data = pd.read_csv(os.path.join(data_dir, csv_file))

    # Create a new figure for the plot
    plt.figure()

    # Create the plot
    plt.plot(data['Time(ms)'], data['Power(mW)'])

    # Set the title and labels
    plt.title(csv_file)
    plt.xlabel('Time (ms)')
    plt.ylabel('Power (mW)')

    # Save the figure as a PNG file
    png_file = os.path.splitext(csv_file)[0] + '.png'
    plt.savefig(os.path.join(plot_dir, png_file))

    # Close the figure to free up memory
    plt.close()
