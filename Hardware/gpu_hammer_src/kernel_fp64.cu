#include "cuda_runtime.h"
#include "cuda_fp16.h"

#define BLOCK_SIZE     640


static __device__ float sfma_out;
static __device__ double dfma_out;

static __global__ void fp64_hammer_kernel()
{
    float sa = -1e5f;
    float sb = -1e5f;
    double da = -1e7;
    double db = -1e7;
    __syncthreads();
    int mag = 2;
    if(BLOCK_SIZE >= 256){
        mag = 1;
    }
    for (int it = 0; it < 102400000 * mag; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm ("fma.rn.f64    %0, %0, 0.9999, 0.01;" : "+d"(da));
            asm ("fma.rn.f64    %0, %0, 0.9999, 0.01;" : "+d"(db));
        }
    }
    if (sa < 0) {
        // This is for avoiding compiler optimization.
        // Program never reach here.
        sfma_out = sa + sb;
        dfma_out = da + db;
    }
}

extern "C" {

cudaError_t fp64_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)fp64_hammer_kernel, grid, block, 0, 256, s);
}

} // extern "C"