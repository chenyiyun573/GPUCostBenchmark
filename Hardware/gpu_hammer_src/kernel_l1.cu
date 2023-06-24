#include "cuda_runtime.h"

// Should be multiple of 32 and power of 2
#define CACHED_ARRAY_SIZE  98304 //96KB
// #define CACHED_ARRAY_SIZE  196608 // 192KB 
#define BLOCK_SIZE     640
static __device__ char arr[CACHED_ARRAY_SIZE];

static __global__ void l1_ld_hammer_kernel()
{
    constexpr int ntmo = BLOCK_SIZE - 1;
    constexpr int nd = CACHED_ARRAY_SIZE / 8;
    double x = 0;
    int tid = threadIdx.x;
    for (int it = 0; it < 6000000; ++it) {
        double *ptr = (double *)arr;
        for (int i = 0; i < nd; i += BLOCK_SIZE) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j += 32) {
                int offset = (tid + j) & ntmo;
                asm volatile ("{\t\n"
                    ".reg .f64 val;\n\t"
                    "ld.global.ca.f64 val, [%1];\n\t"
                    "add.f64 %0, val, %0;\n\t"
                    "}" : "+d"(x) : "l"(ptr+offset) : "memory"
                );
            }
            ptr += 32;
        }
    }
    // For avoiding compiler optimization.
    ((double *)arr)[tid] = x;
}

extern "C" {

cudaError_t l1_ld_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)l1_ld_hammer_kernel, grid, block, 0, 0, s);
}

} // extern "C"
