import threading
import time
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from pynvml import *
import csv

# Initialize NVML
nvmlInit()

# Get the first GPU handle
handle = nvmlDeviceGetHandleByIndex(0)

class TorchRandomDataset(torch.utils.data.Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, *size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class PytorchCNN:
    """The CNN benchmark class."""
    def __init__(self, model_type, batch_size=16, num_classes=1000, sample_count=1024, image_size=224):
        """Constructor.
        Args:
            model_type (str): The cnn benchmark to run.
            batch_size (int): The batch size for training.
            num_classes (int): Number of output classes.
            sample_count (int): Number of samples for the random dataset.
            image_size (int): Size of the image.
        """
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.sample_count = sample_count
        self.image_size = image_size

        # Create a CSV file to store power measurements
        self.power_file = open('power_measurements.csv', 'w', newline='')
        self.power_writer = csv.writer(self.power_file)

        # Write the header row
        self.power_writer.writerow(['Time', 'Power (W)'])

        # Start power monitoring thread
        self.power_monitor_thread = threading.Thread(target=self.monitor_gpu_power)
        self.power_monitor_thread.start()

    def monitor_gpu_power(self):
        start_time = time.time()
        while True:
            try:
                # Get power usage in milliwatts
                power = nvmlDeviceGetPowerUsage(handle)
                
                # Convert to watts
                power = power / 1000.0
                
                current_time = time.time() - start_time
                self.power_writer.writerow([current_time, power])
                
                # Sleep for 0.01 seconds (100 Hz)
                time.sleep(0.01)
                
            except NVMLError as error:
                print(f"Failed to get power usage: {error}")
                break

    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info."""
        dataset = TorchRandomDataset(
            [3, self.image_size, self.image_size],
            self.sample_count
        )
        return DataLoader(dataset, batch_size=self.batch_size)

    def _create_model(self):
        """Construct the model for benchmarking."""
        model = getattr(models, self.model_type)(num_classes=self.num_classes)
        return model

    def run(self):
        dataloader = self._generate_dataset()
        model = self._create_model()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Run training
        for inputs in dataloader:
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            _, _ = torch.max(outputs, 1)
            
            # Assume the labels are random
            labels = torch.randint(0, self.num_classes, (inputs.size(0),))

            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            
            # Optimize
            optimizer.step()

        # Stop power monitoring thread
        self.power_monitor_thread.join()
        self.power_file.close()

# Run benchmark
benchmark = PytorchCNN('resnet50')
benchmark.run()

# Shutdown NVML
nvmlShutdown()
