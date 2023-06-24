import pynvml
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
import os

class PowerMonitor:
    def __init__(self, device_id=0, frequency=100):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.frequency = frequency
        self.power_data = []
        self.stop_flag = False

    def start(self):
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.thread.join()

    def monitor(self):
        while not self.stop_flag:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # convert mW to W
            self.power_data.append(power)
            time.sleep(1.0 / self.frequency)

    def plot(self, name):
        plt.plot(self.power_data)
        plt.xlabel('Time (1/100th second)')
        plt.ylabel('Power (W)')
        plt.savefig('/home/superbench/v-yiyunchen/GPUCostBenchmark/'+name) #TODO

    
    def get_average_power(self, discard_ratio=0.2):
        power_data = self.power_data
        # Determine the indices to discard
        start_discard = int(len(power_data) * discard_ratio)
        end_discard = int(len(power_data) * (1 - discard_ratio))

        # Extract the middle portion of the power data
        middle_power_data = power_data[start_discard:end_discard]

        # Compute and return the average power
        average_power = sum(middle_power_data) / len(middle_power_data)
        return average_power


    def calculate_power_statistics(self, discard_ratio=0.1):
        # Determine the indices to discard
        power_data = self.power_data
        start_discard = int(len(power_data) * discard_ratio)
        end_discard = int(len(power_data) * (1 - discard_ratio))

        # Extract the middle portion of the power data
        middle_power_data = power_data[start_discard:end_discard]

        # Compute statistics
        power_max = np.max(middle_power_data)  # Maximum
        power_min = np.min(middle_power_data)  # Minimum
        power_std = np.std(middle_power_data)  # Standard deviation
        power_mean = np.mean(middle_power_data)  # Mean

        # Prepare the output string
        result_str = f"Mean: {power_mean:.2f}, Max: {power_max:.2f}, Min: {power_min:.2f}, STD: {power_std:.2f}"

        return result_str
    
    def save_log(self, result_string):
        # Ensure the directory exists
        results_dir = './results'
        os.makedirs(results_dir, exist_ok=True)

        # Path to the markdown file
        file_path = os.path.join(results_dir, 'model_results.md')

        # Write (append) the result string to the file
        with open(file_path, 'a') as file:
            file.write(result_string + '\n')


    
    


