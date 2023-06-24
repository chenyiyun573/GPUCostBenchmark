#include "cuda_runtime.h"

#define CACHED_ARRAY_SIZE  49152    // 48KB
#define BLOCK_SIZE     640

static __global__ void smem_ld_hammer_kernel()
{
    __shared__ char arr[CACHED_ARRAY_SIZE];
    constexpr int ntmo = BLOCK_SIZE - 1;
    constexpr int nd = CACHED_ARRAY_SIZE / 8;
    double x = 0;
    int tid = threadIdx.x;
    for (int it = 0; it < 12000000; ++it) {
        double *ptr = (double *)arr;
        for (int i = 0; i < nd; i += BLOCK_SIZE) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j += 32) {
                int offset = (tid + j) & ntmo;
                x += ptr[offset];
            }
            ptr += 32;
        }
    }
    // For avoiding compiler optimization.
    ((double *)arr)[tid] = x;
}

extern "C" {

cudaError_t smem_ld_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)smem_ld_hammer_kernel, grid, block, 0, 0, s);
}

} // extern "C"
