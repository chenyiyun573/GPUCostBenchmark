#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nvml.h>
#include <stdio.h>
#include <pthread.h>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <fstream>


#define BLOCK_SIZE 640
#define SAMPLING_FREQUENCY 50 // Hz

static __device__ float fma_out;

struct PowerSamplingData {
    bool continueSampling;
    std::vector<unsigned int> powerData;
    nvmlDevice_t device;
    std::chrono::milliseconds samplingInterval;
};

void* powerSamplingThreadFunc(void* arg)
{
    PowerSamplingData* data = (PowerSamplingData*)arg;
    while (data->continueSampling) {
        unsigned int power;
        nvmlDeviceGetPowerUsage(data->device, &power);
        data->powerData.push_back(power);
        usleep(data->samplingInterval.count() * 1000);  // convert to microseconds
    }
    return nullptr;
}

extern "C"
{
    __global__ void fp32_hammer_kernel()
    {
        float a = -1e5f;
        float b = -1e5f;
        int mag = 1;
        if (BLOCK_SIZE <= 128)
            mag = 14;
        else if (BLOCK_SIZE <= 256)
            mag = 7;
        else if (BLOCK_SIZE <= 384)
            mag = 5;
        else if (BLOCK_SIZE <= 512)
            mag = 4;
        else
            mag = 3;
        for (int iit = 0; iit < mag; ++iit)
        {
            for (int it = 0; it < 5120000; ++it)
            {
    #pragma unroll
                for (int i = 0; i < 256; ++i)
                {
                    a = __fmaf_ru(a, 0.9999f, 0.01f); // 2 single FLOP
                    b = __fmaf_ru(b, 0.9999f, 0.01f); // 2 single FLOP
                }
            }
        }
        if (a < 0)
        {
            // This is for avoiding compiler optimization.
            // Program never reach here.
            fma_out = a + b;
        }
    }

    cudaError_t fp32_hammer(cudaStream_t s, int nblks)
    {
        dim3 grid = dim3(nblks, 1, 1);
        dim3 block = dim3(BLOCK_SIZE, 1, 1);
        return cudaLaunchKernel((void *)fp32_hammer_kernel, grid, block, 0, 0, s);
    }
}

int main()
{
    // Set the number of blocks you want to launch
    int numBlocks = 1;

    // Initialize NVML
    nvmlReturn_t nvmlResult;
    nvmlResult = nvmlInit();
    if (nvmlResult != NVML_SUCCESS)
    {
        fprintf(stderr, "Failed to initialize NVML: %s\n", nvmlErrorString(nvmlResult));
        return 1;
    }

    // Query GPU device information
    nvmlDevice_t device;
    nvmlResult = nvmlDeviceGetHandleByIndex(0, &device); // Assuming one GPU is available
    if (nvmlResult != NVML_SUCCESS)
    {
        fprintf(stderr, "Failed to get GPU device handle: %s\n", nvmlErrorString(nvmlResult));
        nvmlShutdown();
        return 1;
    }

    // Allocate memory on the device for the workload
    cudaError_t cudaStatus;
    cudaStream_t stream;
    cudaStatus = cudaStreamCreate(&stream);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaStreamCreate failed: %s\n", cudaGetErrorString(cudaStatus));
        nvmlShutdown();
        return 1;
    }

    // Initialize power sampling data
    PowerSamplingData powerSamplingData;
    powerSamplingData.continueSampling = true;
    powerSamplingData.device = device;
    powerSamplingData.samplingInterval = std::chrono::milliseconds(1000 / SAMPLING_FREQUENCY);

    // Create a new thread for power sampling
    pthread_t powerSamplingThread;
    pthread_create(&powerSamplingThread, NULL, powerSamplingThreadFunc, &powerSamplingData);

    // Keep sampling power for an additional 10 seconds before kernel execution
    sleep(8);

    // Launch the workload kernel
    cudaStatus = fp32_hammer(stream, numBlocks);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "fp32_hammer failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStreamDestroy(stream);
        nvmlShutdown();
        return 1;
    }

    // Synchronize the CUDA stream to ensure the kernel execution is completed
    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaStreamSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStreamDestroy(stream);
        nvmlShutdown();
        return 1;
    }

    // Keep sampling power for an additional 10 seconds after kernel execution
    sleep(8);

    // Stop the power sampling thread
    powerSamplingData.continueSampling = false;
    pthread_join(powerSamplingThread, NULL);

    // Open CSV file
    std::ofstream outFile("power_data.csv");
    outFile << "Time(ms),Power(mW)\n";

    // Output the collected power data to the CSV file
    for (size_t i = 0; i < powerSamplingData.powerData.size(); ++i)
    {
        outFile << i * powerSamplingData.samplingInterval.count() << ","
                << powerSamplingData.powerData[i] << "\n";
    }

    // Close the CSV file
    outFile.close();

    // Free the allocated memory, destroy the CUDA stream, and shut down NVML
    cudaStreamDestroy(stream);
    nvmlShutdown();

    return 0;
}
