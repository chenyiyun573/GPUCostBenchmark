#include "cuda_runtime.h"
#include "cuda_fp16.h"

#define BLOCK_SIZE     640

static __device__ float sfma_out;
static __device__ double dfma_out;

static __global__ void fp_hammer_kernel()
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
        for (int it = 0; it < 1024 * 128; ++it)
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

extern "C"
{

    cudaError_t fp_hammer(cudaStream_t s, int nblks)
    {
        dim3 grid = dim3(nblks, 1, 1);
        dim3 block = dim3(BLOCK_SIZE, 1, 1);
        return cudaLaunchKernel((void *)fp_hammer_kernel, grid, block, 0, 256, s);
    }

} // extern "C"