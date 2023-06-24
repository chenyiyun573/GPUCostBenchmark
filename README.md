# GPU Power Benchmark

## Introduction
The GPU Power Benchmark repository is a tool for assessing the power efficiency of GPUs. It is capable of providing performance and power consumption data under various AI workloads and GPU stress tests.

## Prerequisites
Ensure you have the following installed:
- Python 3.x
- NVIDIA CUDA Toolkit (if you wish to compile custom hardware workloads)

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

### 4. Compile Hardware Workload (Optional)
If you wish to add your own hardware workload, compile the CUDA code under folder `Hardware` by executing:

```sh
nvcc main_loop.cu gpu_hammer_src/kernel_*.cu -o main_loop -lnvidia-ml
```

This command compiles the hardware workload part using the NVIDIA CUDA Compiler (nvcc) and links it with the NVIDIA Management Library.

## Contribute
You can contribute to this project by submitting issues or pull requests. Feel free to improve the code or add new features.

## License
This project is open source and available under the [MIT License](LICENSE).

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)


## Disclaimer

### Environmental Factors
This power benchmark is focused on assessing the GPU's power efficiency and performance. It does not take into account various environmental factors that could affect the results. Specifically, the tool does not consider:

- Environmental temperature
- Airflow
- Air pressure
- Humidity

Additionally, the influence of the power supply, the source of power, and various configurations on the GPU and server are also not accounted for. Users should be aware that these factors can have a significant impact on the performance and power consumption of the GPU.

### Hardware Compatibility
Currently, the GPU Power Benchmark tool supports only NVIDIA GPUs. Compatibility with GPUs from other manufacturers is not available at this time. Users are encouraged to verify the compatibility of their hardware before using this tool.





