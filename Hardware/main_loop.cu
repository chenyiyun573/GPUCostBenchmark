// main.cu
#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <pthread.h>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <fstream>
#include <iostream>


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


// Declare your workloads' interface functions.
extern "C" cudaError_t fp_hammer(cudaStream_t s, int nblks);//0
extern "C" cudaError_t fp32_hammer(cudaStream_t s, int nblks);//1
extern "C" cudaError_t fp64_hammer(cudaStream_t s, int nblks);//2
extern "C" cudaError_t gmem_fp_hammer(cudaStream_t s, int nblks);//3 TODO
extern "C" cudaError_t gmem_ld_hammer(cudaStream_t s, int nblks);//4 TODO
extern "C" cudaError_t l1_ld_hammer(cudaStream_t s, int nblks);//5 
extern "C" cudaError_t l2_ld_hammer(cudaStream_t s, int nblks);//6
extern "C" cudaError_t smem_ld_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t tensor_f16f16f32_hammer(cudaStream_t s, int nblks);//7 TODO
extern "C" cudaError_t tensor_f16f16f16_hammer(cudaStream_t s, int nblks);//8 TODO
extern "C" cudaError_t tensor_bf16bf16f32_hammer(cudaStream_t s, int nblks);//9
extern "C" cudaError_t tensor_tf32tf32f32_hammer(cudaStream_t s, int nblks);//10
extern "C" cudaError_t tensor_f64f64f64_hammer(cudaStream_t s, int nblks);//11
extern "C" cudaError_t tensor_s8s8s32_hammer(cudaStream_t s, int nblks);//12
extern "C" cudaError_t tensor_s4s4s32_hammer(cudaStream_t s, int nblks);//13
extern "C" cudaError_t tensor_b1b1s32_hammer(cudaStream_t s, int nblks);//14
// ... Declare as many workloads as you have ...

int main() {
    // Check if the results directory exists and if not, create it.
    system("mkdir -p ./results/hammer");

    // Initialization
    // Initialize NVML
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);

    // Initialize power sampling data
    PowerSamplingData powerSamplingData;
    powerSamplingData.continueSampling = true;
    powerSamplingData.device = device;
    powerSamplingData.samplingInterval = std::chrono::milliseconds(1000 / SAMPLING_FREQUENCY);

    // Create a vector of function pointers to your workloads' interface functions.
    std::vector<cudaError_t (*)(cudaStream_t, int)> workloads = {
        
        fp_hammer,
        //fp32_hammer,
        //fp64_hammer,
        
        //gmem_fp_hammer,
        //gmem_ld_hammer,
        
        //l1_ld_hammer,
        //l2_ld_hammer,
        
        //smem_ld_hammer,
        //tensor_f16f16f32_hammer,
        //tensor_f16f16f16_hammer,
        //tensor_bf16bf16f32_hammer,
        //tensor_tf32tf32f32_hammer,
        //tensor_f64f64f64_hammer,
        //tensor_s8s8s32_hammer,
        //tensor_s4s4s32_hammer,
        //tensor_b1b1s32_hammer
    };

    // Define powerSamplingThread
    pthread_t powerSamplingThread;  // Add this line

    int workloadNum = 0;
    for (auto& workload : workloads) {
        // Print update to terminal
        std::cout << "Running workload number: " << workloadNum + 1 << std::endl;

        // Initialize power sampling data and start the powerSamplingThread
        powerSamplingData.continueSampling = true;
        powerSamplingData.powerData.clear();
        pthread_create(&powerSamplingThread, NULL, powerSamplingThreadFunc, &powerSamplingData);

        // Keep sampling power for an additional 8 seconds before kernel execution
        sleep(8);

        // Call kernel
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        workload(stream, 1024);  // Call the current workload.
        cudaDeviceSynchronize();

        // Keep sampling power for an additional 8 seconds after kernel execution
        sleep(8);

        // Stop power sampling
        powerSamplingData.continueSampling = false;
        pthread_join(powerSamplingThread, NULL);

        // Output to file
        std::string filename = "./results/hammer/powerData_" + std::to_string(workloadNum) + ".csv";
        std::ofstream file(filename);
        file << "Time(ms),Power(mW)\n";
        for (int i = 0; i < powerSamplingData.powerData.size(); i++) {
            file << i * (1000 / SAMPLING_FREQUENCY) << "," << powerSamplingData.powerData[i] << "\n";
        }
        file.close();

        // Reset for next workload
        powerSamplingData.continueSampling = true;
        powerSamplingData.powerData.clear();
        workloadNum++;

        // Print completion update to terminal
        std::cout << "Completed workload number: " << workloadNum << std::endl;
    }


    nvmlShutdown();
    return 0;
}
