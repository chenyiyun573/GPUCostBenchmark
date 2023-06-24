#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <pthread.h>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <fstream>
#include <future>

#define CACHED_ARRAY_SIZE  49152    // 48KB
#define BLOCK_SIZE     640
#define SAMPLING_FREQUENCY 50 // Hz
#define TIME_LIMIT std::chrono::seconds(20)  // Timeout in seconds

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

__global__ void smem_ld_hammer_kernel()
{
    __shared__ char arr[CACHED_ARRAY_SIZE];
    constexpr int ntmo = BLOCK_SIZE - 1;
    constexpr int nd = CACHED_ARRAY_SIZE / 8;
    double x = 0;
    int tid = threadIdx.x;
    for (int it = 0; it < 12000000; ++it) {
        double *ptr = (double *)arr;
        for (int i = 0; i < nd; i += BLOCK_SIZE) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j += 32) {
                int offset = (tid + j) & ntmo;
                x += ptr[offset];
            }
            ptr += 32;
        }
    }
    ((double *)arr)[tid] = x;
}

extern "C" {

cudaError_t smem_ld_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)smem_ld_hammer_kernel, grid, block, 0, 0, s);
}

} // extern "C"

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

    // Start the power sampling thread
    PowerSamplingData samplingData;
    samplingData.continueSampling = true;
    samplingData.device = device;
    samplingData.samplingInterval = std::chrono::milliseconds(1000 / SAMPLING_FREQUENCY);
    pthread_t samplingThread;
    pthread_create(&samplingThread, NULL, powerSamplingThreadFunc, &samplingData);

    sleep(8);

    // Run the workload in a separate thread, with a timeout
    std::future<cudaError_t> fut = std::async(std::launch::async, smem_ld_hammer, stream, numBlocks);
    if(fut.wait_for(TIME_LIMIT) == std::future_status::timeout) {
        // Workload has exceeded time limit. Cleaning up.
        cudaStreamDestroy(stream);
        samplingData.continueSampling = false;
        pthread_join(samplingThread, NULL);
        nvmlShutdown();
        return 1;
    }
    else {
        // Workload has finished within time limit
        cudaStatus = fut.get();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "smem_ld_hammer failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        sleep(8);

        cudaStreamDestroy(stream);
        samplingData.continueSampling = false;
        pthread_join(samplingThread, NULL);
        nvmlShutdown();

        

        // Write power data to file
        std::ofstream outputFile("power_data.csv");
        outputFile << "Time(ms),Power(mW)\n";
        for (size_t i = 0; i < samplingData.powerData.size(); ++i)
        {
            outputFile << i * samplingData.samplingInterval.count() << ","
                    << samplingData.powerData[i] << "\n";
        }
        outputFile.close();

        return 0;
    }
}
