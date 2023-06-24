#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nvml.h>
#include <stdio.h>
#include <pthread.h>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <fstream>

#define ARRAY_SIZE 268435456   // 256M
#define BLOCK_SIZE 1024
#define UNROLL_DEPTH 128
#define SAMPLING_FREQUENCY 50 // Hz

static __device__ char ld_arr[ARRAY_SIZE];
static __device__ float sfma_out;
static __device__ double dfma_out;

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
    __global__ void gmem_fp_hammer_kernel()
    {
        int nd = ARRAY_SIZE >> 3;
        volatile double *ptr = (volatile double *)ld_arr;
        int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
        int sz = BLOCK_SIZE * gridDim.x;
        int usz = sz * UNROLL_DEPTH;
        double x = 0;

        float sa = -1e5f;
        float sb = -1e5f;
        double da = -1e7;
        double db = -1e7;

        nd -= (nd % usz);
        __syncthreads();
        for (int it = 0; it < 1024; ++it) {
            for (int i = idx; i < nd; i += usz) {
                #pragma unroll
                for (int j = 0; j < UNROLL_DEPTH; ++j) {
                    x += ptr[i+j*sz];
                }
                #pragma unroll
                for (int j = 0; j < 670; ++j) {
                    asm ("fma.rn.f64    %0, %0, 0.9999, 0.01;" : "+d"(da));
                    asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sa));
                    asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sb));
                    asm ("fma.rn.f64    %0, %0, 0.9999, 0.01;" : "+d"(db));
                    asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sa));
                    asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sb));
                }
            }
        }
        if (sa < 0) {
            // This is for avoiding compiler optimization.
            // Program never reach here.
            sfma_out = sa + sb;
            dfma_out = da + db;
            ptr[idx] = x;
        }
    }

    cudaError_t gmem_fp_hammer(cudaStream_t s, int nblks)
    {
        dim3 grid = dim3(nblks, 1, 1);
        dim3 block = dim3(BLOCK_SIZE, 1, 1);
        return cudaLaunchKernel((void *)gmem_fp_hammer_kernel, grid, block, 0, 0, s);
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
    cudaStatus = gmem_fp_hammer(stream, numBlocks);
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

