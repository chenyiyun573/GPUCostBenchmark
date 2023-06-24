#include "cuda_runtime.h"

// Should be multiple of 32 and power of 2
// TODO: kernel crashes when CACHED_ARRAY_SIZE=64KB
#define CACHED_ARRAY_SIZE 98304 //96KB
#define BLOCK_SIZE     640
static __device__ char arr[CACHED_ARRAY_SIZE];

// Referred to code in https://arxiv.org/pdf/1804.06826.pdf
static __global__ void l2_ld_hammer_kernel()
{
    constexpr int ntmo = BLOCK_SIZE - 1;
    constexpr int nd = CACHED_ARRAY_SIZE / 8;
    double x = 0;
    int tid = threadIdx.x;
    int mag = 1;

    // if (BLOCK_SIZE <= 64) //64: 60.3s
    // {
    //     mag = 30;
    // }
    // else if (BLOCK_SIZE <= 128) // 128: 56.8s
    // {
    //     mag = 15;
    // }
    // else if (BLOCK_SIZE <= 192) // 192: 45.6s
    // {
    //     mag = 6;
    // }
    // else if (BLOCK_SIZE <= 320) // 46.3s
    // {
    //     mag = 3;
    // }
    // else if (BLOCK_SIZE <= 512) // 512: 35.4s
    // {
    //     mag = 1.5;
    // }
    // // 640: 42.6s
    // else {
    //     mag  = 1;
    // }
    for (int it = 0; it < 120000 * mag; ++it)
    {
        double *ptr = (double *)arr;
        for (int i = 0; i < nd; i += BLOCK_SIZE)
        {
#pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j += 32)
            {
                int offset = (tid + j) & ntmo;
                asm volatile("{\t\n"
                             ".reg .f64 val;\n\t"
                             "ld.global.cg.f64 val, [%1];\n\t"
                             "add.f64 %0, val, %0;\n\t"
                             "}"
                             : "+d"(x)
                             : "l"(ptr + offset)
                             : "memory");
            }
            ptr += 32;
        }
    }
    // For avoiding compiler optimization.
    ((double *)arr)[tid] = x;
}

extern "C"
{

    cudaError_t l2_ld_hammer(cudaStream_t s, int nblks)
    {
        dim3 grid = dim3(nblks, 1, 1);
        dim3 block = dim3(BLOCK_SIZE, 1, 1);
        return cudaLaunchKernel((void *)l2_ld_hammer_kernel, grid, block, 0, 0, s);
    }

} // extern "C"
