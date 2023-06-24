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

extern "C" cudaError_t gmem_ld_hammer(cudaStream_t s, int nblks);//3 TODO

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
    gmem_ld_hammer(stream, 1024);
    cudaDeviceSynchronize();

    // Keep sampling power for an additional 8 seconds after kernel execution
    sleep(2);

    // Stop power sampling
    powerSamplingData.continueSampling = false;
    pthread_join(powerSamplingThread, NULL);

    // Output to file
    // Check if the results directory exists and if not, create it.
    system("mkdir -p results");
    std::ofstream file("results/powerData.csv");
    file << "Time(ms),Power(mW)\n";
    for (int i = 0; i < powerSamplingData.powerData.size(); i++) {
        file << i * (1000 / SAMPLING_FREQUENCY) << "," << powerSamplingData.powerData[i] << "\n";
    }
    file.close();

    nvmlShutdown();
    return 0;
}
