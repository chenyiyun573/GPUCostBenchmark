#include "cuda_runtime.h"
#include "cuda_fp16.h"

#define BLOCK_SIZE     640

static __device__ float sfma_out;
static __device__ double dfma_out;

static __global__ void fp_hammer_kernel()
{
    double da = -1e7;
    double db = -1e7;
    int mag = 2;
    __syncthreads();
    
    for (int iit = 0; iit < mag; ++iit)
    {
        for (int it = 0; it < 1024*512; ++it)
        {
#pragma unroll
            for (int i = 0; i < 32; ++i)
            {
                asm("add.f64 %0, %0, 0.01;"
                    : "+d"(da));
                asm("add.f64 %0, %0, 0.01;"
                    : "+d"(db));
                asm("add.f64 %0, %0, 0.01;"
                    : "+d"(da));
                asm("add.f64 %0, %0, 0.01;"
                    : "+d"(db));
            }
        }
    }
    
    if (da < 0)
    {
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
