#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nvml.h>
#include <stdio.h>

#define BLOCK_SIZE 640

static __device__ float sfma_out;
static __device__ double dfma_out;



extern "C"
{
    __global__ void fp_hammer_kernel()
    {
        float sa = -1e5f;
        float sb = -1e5f;
        double da = -1e7;
        double db = -1e7;
        int mag = 0;
        __syncthreads();
        if (BLOCK_SIZE <= 128) //128 - 39.79018793s
            mag = 8;
        else if (BLOCK_SIZE <= 256) //256 - 46.756912947s
            mag = 5;
        else if (BLOCK_SIZE <= 384) //384 - 55.8s
            mag = 4;
        else if (BLOCK_SIZE <= 512) //512 - 56s
            mag = 3;
        else // 640 - 46.871180901s
            mag = 2;
        for (int iit = 0; iit < mag; ++iit)
        {
            for (int it = 0; it < 102400 * 200; ++it)
            {
    #pragma unroll
                for (int i = 0; i < 32; ++i)
                {
                    asm("fma.rn.f64    %0, %0, 0.9999, 0.01;"
                        : "+d"(da));
                    asm("fma.rn.f64    %0, %0, 0.9999, 0.01;"
                        : "+d"(db));
                    asm("fma.rn.f32    %0, %0, 0.9999, 0.01;"
                        : "+f"(sa));
                    asm("fma.rn.f32    %0, %0, 0.9999, 0.01;"
                        : "+f"(sb));
                }
            }
        }
        if (sa < 0)
        {
            // This is for avoiding compiler optimization.
            // Program never reach here.
            sfma_out = sa + sb;
            dfma_out = da + db;
        }
    }

    cudaError_t fp_hammer(cudaStream_t s, int nblks)
    {
        dim3 grid(nblks, 1, 1);
        dim3 block(BLOCK_SIZE, 1, 1);
        fp_hammer_kernel<<<grid, block, 0, s>>>();
        return cudaGetLastError();
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

    // Launch the workload kernel
    cudaStatus = fp_hammer(stream, numBlocks);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "fp_hammer failed: %s\n", cudaGetErrorString(cudaStatus));
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

    // Query GPU power consumption using NVML
    unsigned int power;
    nvmlResult = nvmlDeviceGetPowerUsage(device, &power);
    if (nvmlResult != NVML_SUCCESS)
    {
        fprintf(stderr, "Failed to get GPU power usage: %s\n", nvmlErrorString(nvmlResult));
        cudaStreamDestroy(stream);
        nvmlShutdown();
        return 1;
    }

    printf("GPU Power Usage: %u mW\n", power);

    // Free the allocated memory, destroy the CUDA stream, and shut down NVML
    cudaStreamDestroy(stream);
    nvmlShutdown();

    return 0;
}
