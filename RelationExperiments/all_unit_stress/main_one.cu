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

extern "C" cudaError_t stress_units(cudaStream_t s, int nblks, int unroll_num);

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
    for(int unroll_num=1; unroll_num <= 128*8; unroll_num *= 2) {
        int nblks = 1024;

        // Reset power data and continue sampling flag for each iteration
        powerSamplingData.powerData.clear();
        powerSamplingData.continueSampling = true;

        // Keep sampling power for an additional 2 seconds before kernel execution
        sleep(2);

        // Call kernel
        stress_units(stream, nblks, unroll_num);
        cudaDeviceSynchronize();

        // Keep sampling power for an additional 2 seconds after kernel execution
        sleep(2);

        // Stop power sampling
        powerSamplingData.continueSampling = false;

        // Output to file
        // Check if the results directory exists and if not, create it.
        system("mkdir -p results");
        std::ofstream file("results/powerData_" + std::to_string(unroll_num) + ".csv");
        file << "Time(ms),Power(mW)\n";
        for (int i = 0; i < powerSamplingData.powerData.size(); i++) {
            file << i * (1000 / SAMPLING_FREQUENCY) << "," << powerSamplingData.powerData[i] << "\n";
        }
        file.close();
    }

    // Join the power sampling thread after all kernel launches
    pthread_join(powerSamplingThread, NULL);

    nvmlShutdown();
    return 0;
}
