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
#include <cmath> 

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
        
        //fp_hammer,
        fp32_hammer,  //15728640000 * 1024(nblk) = 16106127360000 FLO 
        fp64_hammer,  //13107200000 * 1024(nblk) = 13421772800000 FLO (one FMA will be regarded as two FLO)
        
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
    pthread_t powerSamplingThread;

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
    
    std::ofstream markdownFile("./results/hammer.md");


    markdownFile << "| Workload Name   | Total Time (s) | FLOPs | FLOPs/Watt | STD of Power (W) | Mean Power (W) | Range of Power (W) |\n";
    markdownFile << "|-----------------|----------------|-------|------------|------------------|----------------|--------------------|\n";

    std::vector<double> total_flops_list = {16106127360000.0, 13421772800000.0, /* ... other FLOPs values for other workloads ... */};
    std::vector<std::string> workload_names = {
        "fp32_hammer",
        "fp64_hammer"
        // add names for other workloads
    };

    for (int workloadIndex = 0; workloadIndex < workloadNum; workloadIndex++) {
        std::string filename = "./results/hammer/powerData_" + std::to_string(workloadIndex) + ".csv";
        std::ifstream file(filename);

        std::vector<double> powerData;
        std::string line;
        std::getline(file, line); // Skip header line
        while (std::getline(file, line)) {
            unsigned int power_mW = std::stoi(line.substr(line.find(",") + 1));
            powerData.push_back(static_cast<double>(power_mW) / 1000); // Convert to Watts
        }
        file.close();

        // Calculate metrics
        double total_time = (powerData.size() / static_cast<double>(SAMPLING_FREQUENCY)) - 16; // Subtract sleep time before and after kernel execution
        double flops = total_flops_list[workloadIndex] / total_time;
        
        int trimSize = powerData.size() * 0.2;
        double sum = 0.0, sumSq = 0.0, maxPower = 0.0, minPower = std::numeric_limits<double>::max();

        for (int i = trimSize; i < powerData.size() - trimSize; i++) {
            sum += powerData[i];
            sumSq += powerData[i] * powerData[i];
            maxPower = std::max(maxPower, powerData[i]);
            minPower = std::min(minPower, powerData[i]);
        }
        
        double mean = sum / (powerData.size() - 2 * trimSize);
        double stdDev = std::sqrt((sumSq / (powerData.size() - 2 * trimSize)) - (mean * mean));
        double flops_per_watt = total_flops_list[workloadIndex] / (total_time * mean);

        // Output metrics to markdown file
        markdownFile << "| " << workload_names[workloadIndex] << " | ";
        markdownFile << total_time << "          | ";
        markdownFile << flops << " | ";
        markdownFile << flops_per_watt << " | ";
        markdownFile << stdDev << "        | ";
        markdownFile << mean << "        | ";
        markdownFile << maxPower << " - " << minPower << " |\n";
    }

    

    markdownFile.close();


    nvmlShutdown();


    return 0;
}
