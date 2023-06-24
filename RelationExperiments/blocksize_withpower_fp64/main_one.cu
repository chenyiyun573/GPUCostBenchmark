// main.cu
#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <pthread.h>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <fstream>

#define SAMPLING_FREQUENCY 10 // 10Hz
#define FLOPS_PER_THREAD 33554432LL // Calculated from kernel

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

extern "C" cudaError_t fp_hammer(cudaStream_t s, int nblks);

int main() {
    // Initialize NVML
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Initialize power sampling data
    PowerSamplingData powerSamplingData;
    powerSamplingData.device = device;
    powerSamplingData.samplingInterval = std::chrono::milliseconds(1000 / SAMPLING_FREQUENCY);

    // Create a new thread for power sampling
    pthread_t powerSamplingThread;
    pthread_create(&powerSamplingThread, NULL, powerSamplingThreadFunc, &powerSamplingData);

    // Loop over number of blocks
    for(int nblks = 2; nblks <= 1024*64; nblks *= 2) {
        // Reset power data and continue sampling flag for each iteration
        powerSamplingData.powerData.clear();
        powerSamplingData.continueSampling = true;

        // Keep sampling power for an additional 2 seconds before kernel execution
        sleep(2);

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Call kernel
        fp_hammer(stream, nblks);
        cudaDeviceSynchronize();

        // Stop timing
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        // Keep sampling power for an additional 2 seconds after kernel execution
        sleep(2);

        // Stop power sampling
        powerSamplingData.continueSampling = false;

        // Calculate average power
        // Calculate average power
        unsigned int sumPower = 0;
        int powerDataSize = powerSamplingData.powerData.size();
        int numSamplesToSkip = 2 * SAMPLING_FREQUENCY; // Number of samples during 2 seconds
        for (int i = numSamplesToSkip; i < powerDataSize - numSamplesToSkip; i++)
            sumPower += powerSamplingData.powerData[i];
        double avgPower = static_cast<double>(sumPower) / (powerDataSize - 2 * numSamplesToSkip) / 1000; // convert to watts

        double tflops = (static_cast<double>(FLOPS_PER_THREAD * nblks * 640)) / (duration.count() * 1e6); // TFLOPs



        printf("nblks: %i, Execution Time: %ld us, Performance: %lf TFLOPS, Average Power: %lf W, Power Efficiency: %lf TFLOPS/W\n", nblks,
                duration.count(), tflops, avgPower, tflops / avgPower);
        
        
    }

    // Join the power sampling thread after all kernel launches
    pthread_join(powerSamplingThread, NULL);

    nvmlShutdown();
    return 0;
}
