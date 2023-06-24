#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <pthread.h>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <fstream>

#define CACHED_ARRAY_SIZE  49152    // 48KB
#define BLOCK_SIZE     640
#define SAMPLING_FREQUENCY 10 // 10Hz

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

static __global__ void smem_ld_hammer_kernel()
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
    // For avoiding compiler optimization.
    ((double *)arr)[tid] = x;
}

extern "C" {

cudaError_t smem_ld_hammer(cudaStream_t s, int nblks)
{
    dim3 grid(nblks, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    smem_ld_hammer_kernel<<<grid, block, 0, s>>>();
    return cudaGetLastError();
}

} // extern "C"

int main() {
    // Initialize NVML
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);

    // Initialize power sampling data
    PowerSamplingData powerSamplingData;
    powerSamplingData.continueSampling = true;
    powerSamplingData.device = device;
    powerSamplingData.samplingInterval = std::chrono::milliseconds(1000 / SAMPLING_FREQUENCY);

    // Create a new thread for power sampling
    pthread_t powerSamplingThread;
    pthread_create(&powerSamplingThread, NULL, powerSamplingThreadFunc, &powerSamplingData);

    // Keep sampling power for an additional 8 seconds before kernel execution
    sleep(8);

    // Call kernel
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    smem_ld_hammer(stream, 1024);
    cudaDeviceSynchronize();

    // Keep sampling power for an additional 8 seconds after kernel execution
    sleep(16);

    // Stop power sampling
    powerSamplingData.continueSampling = false;
    pthread_join(powerSamplingThread, NULL);

    // Output to file
    std::ofstream file("powerData.csv");
    file << "Time(ms),Power(mW)\n";
    for (int i = 0; i < powerSamplingData.powerData.size(); i++) {
        file << i * (1000 / SAMPLING_FREQUENCY) << "," << powerSamplingData.powerData[i] << "\n";
    }
    file.close();

    nvmlShutdown();
    return 0;
}
