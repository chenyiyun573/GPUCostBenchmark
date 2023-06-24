#include "cuda_runtime.h"

static __device__ float fma_out;
#define BLOCK_SIZE     640

static __global__ void fp32_hammer_kernel()
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

extern "C"
{

    cudaError_t fp32_hammer(cudaStream_t s, int nblks)
    {
        dim3 grid = dim3(nblks, 1, 1);
        dim3 block = dim3(BLOCK_SIZE, 1, 1);
        return cudaLaunchKernel((void *)fp32_hammer_kernel, grid, block, 0, 0, s);
    }

} // extern "C"
