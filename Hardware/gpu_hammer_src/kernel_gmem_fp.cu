#include "cuda_runtime.h"
#include "cuda_fp16.h"

// Should be multiple of 32
#define ARRAY_SIZE      268435456   // 256M
// Should be multiple of 32 and power of 2
#define BLOCK_SIZE     1024
//
#define UNROLL_DEPTH    128

static __device__ char ld_arr[ARRAY_SIZE];

static __device__ float sfma_out;
static __device__ double dfma_out;

static __global__ void gmem_fp_hammer_kernel()
{
    int nd = ARRAY_SIZE >> 3;
    volatile double *ptr = (volatile double *)ld_arr;
    int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    int sz = BLOCK_SIZE * gridDim.x;
    int usz = sz * UNROLL_DEPTH;
    double x = 0;

    float sa = -1e5f;
    float sb = -1e5f;
    double da = -1e7;
    double db = -1e7;

    nd -= (nd % usz);
    __syncthreads();
    for (int it = 0; it < 1024; ++it) {
        for (int i = idx; i < nd; i += usz) {
            #pragma unroll
            for (int j = 0; j < UNROLL_DEPTH; ++j) {
                x += ptr[i+j*sz];
            }
            #pragma unroll
            for (int j = 0; j < 670; ++j) {
                asm ("fma.rn.f64    %0, %0, 0.9999, 0.01;" : "+d"(da));
                asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sa));
                asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sb));
                asm ("fma.rn.f64    %0, %0, 0.9999, 0.01;" : "+d"(db));
                asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sa));
                asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sb));
            }
        }
    }
    if (sa < 0) {
        // This is for avoiding compiler optimization.
        // Program never reach here.
        sfma_out = sa + sb;
        dfma_out = da + db;
        ptr[idx] = x;
    }
}

extern "C" {

cudaError_t gmem_fp_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)gmem_fp_hammer_kernel, grid, block, 0, 0, s);
}

} // extern "C"
