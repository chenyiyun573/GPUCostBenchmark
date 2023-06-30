# GPU Power Benchmark

## Introduction
The GPU Power Benchmark repository is a tool for assessing the power efficiency of GPUs. It is capable of providing performance and power consumption data under various AI workloads and GPU stress tests.

This repo has not been completed yet as a comprehensive benchmark.(20230630) But all code can work now. The gpu hammer src kernels have not been verified or modified for benchmarking now. 

## Prerequisites
Ensure you have the following installed:
- Python 3.x >= 3.7
- NVIDIA CUDA Toolkit (if you wish to compile custom hardware workloads)
Recommend CUDA11.7, and there are some errors to install python packages using old cuda toolkit. 

## Getting Started
Follow these steps to get the GPU Power Benchmark tool up and running on your system.

### 1. Clone the Repository
Start by cloning the repository to your local machine. Use the following command:

```sh
git clone <REPOSITORY_URL>
```

Replace `<REPOSITORY_URL>` with the URL of this repository.

### 2. Install Dependencies
Navigate to the repository directory and install the required Python dependencies with the following command:

```sh
pip install -r requirements.txt
```

### 3. Execute the Benchmark
Run the benchmark using Python by executing:

```sh
python3 benchmark.py
```

Replace `benchmark.py` with the appropriate script name if different.


### 4. Choose Models in Application Workload (Optional)


### 4. Compile Hardware Workload (Optional)
Currently, only fp32 and fp64 are supported and FLOPS metric is provided only for these two workload. 

If you wish to add your own hardware workload, compile the CUDA code under folder `Hardware` by executing:

```sh
nvcc main_loop.cu gpu_hammer_src/kernel_*.cu -o main_loop -lnvidia-ml
```

This command compiles the hardware workload part using the NVIDIA CUDA Compiler (nvcc) and links it with the NVIDIA Management Library.


### 5. Result
All results will be output into the folder `./results/`.
1. model_results.md is for models trainning or inference with step/(s*watt) as the Power Metric
2. hammer.md is the FLOPS/watt metric for Hardware workload.
3. See `./results/idle` for idle measurement results. 

## Disclaimer

### Tasks on Multiple GPUs
This power benchmark does not support distributed training on multiple GPUs or multiple nodes now.
For idle measurement, it will record all GPU's data on the same node/machine.
For workload running, it only record one GPU (index:0 by default) on the node/machine.

### Environmental Factors
This power benchmark is focused on assessing the GPU's power efficiency and performance. It does not take into account various environmental factors that could affect the results. Specifically, the tool does not consider:

- Environmental temperature
- Airflow
- Air pressure
- Humidity

Additionally, the influence of the power supply, the source of power, and various configurations on the GPU and server are also not accounted for. Users should be aware that these factors may have a significant impact on the performance and power consumption of the GPU.

### Hardware Compatibility
Currently, the GPU Power Benchmark tool supports only NVIDIA GPUs. Compatibility with GPUs from other manufacturers is not available at this time. Users are encouraged to verify the compatibility of their hardware before using this tool.

## Power Metric
### For models training and inference:
The sampling frequency is 10Hz in a seperate process.
Start to sample at the beginning of model training or inference iterations, and end just after iterations completed.
We excluded the initial and final 20% of the power data series and focused solely on the middle 60% to calculate the power metric.

Performance: 1000/step time(in ms)
Power Efficiency: Performance/Watt = step/(s*watt)


### For operations stress test:
The sampling frequency is 10Hz in a seperate process.
Sleep 8s before the kernel starts and after the kernel ends. Get samples along kernel running.

Performance: FLOPS
Power Efficiency: FLOPS/watt



## Contribute
You can contribute to this project by submitting issues or pull requests. Feel free to improve the code or add new features.

## License
The application workload part is based on Superbench by Microsoft. 
The hardware stress test part is based on Util_GPUModel by Microsoft.
