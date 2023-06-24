#include <cuda_runtime.h>
#include <mma.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 640

// Using nvcuda::wmma namespace for wmma API
using namespace nvcuda;

static __device__ volatile float sfma_out;
static __device__ volatile double dfma_out;
static __device__ volatile int si32_out;

static __global__ void stress_kernel(int unroll_num)
{
    float sf32 = 0.1f;
    double sd64 = 0.1;
    int si32 = 1;

    // Initialize fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    // Fill fragments
    wmma::fill_fragment(a_frag, 0.1f);
    wmma::fill_fragment(b_frag, 0.1f);
    wmma::fill_fragment(c_frag, 0.0f);

    for (int it = 0; it < 1024 * 512; ++it)
    {
#pragma unroll
        for (int i = 0; i < unroll_num; ++i)
        {
            // INT32 operation
            si32 = si32 * si32;

            // FP64 operation
            sd64 = fma(sd64, sd64, sd64);

            // FP32 operation
            sf32 = fma(sf32, sf32, sf32);

            // Tensor Core operation
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    if (si32 < 0)
    {
        // This is for avoiding compiler optimization.
        // Program never reaches here.
        si32_out = si32;
        sfma_out = sf32;
        dfma_out = sd64;
    }
}

extern "C"
{
    cudaError_t stress_units(int nblks, int unroll_num)
    {
        dim3 grid(nblks, 1, 1);
        dim3 block(BLOCK_SIZE, 1, 1);
        stress_kernel<<<grid, block>>>(unroll_num);
        return cudaGetLastError();
    }
} // extern "C"
