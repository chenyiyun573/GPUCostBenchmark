#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "mma.h"
#define BLOCK_SIZE     640

static __device__ float tensor_out;

static __global__ void tensor_f16f16f16_hammer_kernel()
{
#if (__CUDA_ARCH__ >= 800)
    __shared__ unsigned int smem[3];
    __half2 *A = (__half2 *)&smem[0];
    A[0] = __float2half2_rn(0.01);
    A[1] = __float2half2_rn(0.01);

    __half2 *B = (__half2 *)&smem[2];
    B[0] = __float2half2_rn(0.01);

    unsigned int C0[2];
    unsigned int C1[2];

    *(__half2 *)&C0[0] = __float2half2_rn(0.01);
    *(__half2 *)&C0[1] = __float2half2_rn(0.01);
    *(__half2 *)&C1[0] = __float2half2_rn(0.01);
    *(__half2 *)&C1[1] = __float2half2_rn(0.01);

    __syncthreads();

    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm (
                "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                "   { %0, %1 }, "
                "   { %2, %3 }, "
                "   { %4 },     "
                "   { %0, %1 }; "
                : "+r"(C0[0]), "+r"(C0[1])
                : "r"(smem[0]), "r"(smem[1]),
                  "r"(smem[2])
            );
            asm (
                "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                "   { %0, %1 }, "
                "   { %2, %3 }, "
                "   { %4 },     "
                "   { %0, %1 }; "
                : "+r"(C1[0]), "+r"(C1[1])
                : "r"(smem[0]), "r"(smem[1]),
                  "r"(smem[2])
            );
        }
    }
    if (C0[0] > 0.2) {
        tensor_out = C0[0] + C0[1] + C1[0] + C1[1];
    }
#endif // (__CUDA_ARCH__ >= 800)
}

static __global__ void tensor_f16f16f32_hammer_kernel()
{
#if (__CUDA_ARCH__ >= 800)
    __shared__ unsigned int smem[3];
    __half2 *A = (__half2 *)&smem[0];
    A[0] = __float2half2_rn(0.01);
    A[1] = __float2half2_rn(0.01);

    __half2 *B = (__half2 *)&smem[2];
    B[0] = __float2half2_rn(0.01);

    float C0[4] = { 0.01, 0.01, 0.01, 0.01 };
    float C1[4] = { 0.01, 0.01, 0.01, 0.01 };

    __syncthreads();

    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm (
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5 },         "
                "   { %6 },             "
                "   { %0, %1, %2, %3 }; "
                : "+f"(C0[0]), "+f"(C0[1]), "+f"(C0[2]), "+f"(C0[3])
                : "r"(smem[0]), "r"(smem[1]),
                  "r"(smem[2])
            );
            asm (
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5 },         "
                "   { %6 },             "
                "   { %0, %1, %2, %3 }; "
                : "+f"(C1[0]), "+f"(C1[1]), "+f"(C1[2]), "+f"(C1[3])
                : "r"(smem[0]), "r"(smem[1]),
                  "r"(smem[2])
            );
        }
    }
    if (C0[0] > 0.2) {
        tensor_out =
            C0[0] + C0[1] + C0[2] + C0[3] +
            C1[0] + C1[1] + C1[2] + C1[3];
    }
#endif // (__CUDA_ARCH__ >= 800)
}

static __global__ void tensor_bf16bf16f32_hammer_kernel()
{
#if (__CUDA_ARCH__ >= 800)
    __shared__ unsigned int smem[3];
    __nv_bfloat162 *A = (__nv_bfloat162 *)&smem[0];
    A[0] = __float2bfloat162_rn(0.01);
    A[1] = __float2bfloat162_rn(0.01);

    __nv_bfloat162 *B = (__nv_bfloat162 *)&smem[2];
    B[0] = __float2bfloat162_rn(0.01);

    float C0[4] = { 0.01, 0.01, 0.01, 0.01 };
    float C1[4] = { 0.01, 0.01, 0.01, 0.01 };

    __syncthreads();

    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm (
                "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5 },         "
                "   { %6 },             "
                "   { %0, %1, %2, %3 }; "
                : "+f"(C0[0]), "+f"(C0[1]), "+f"(C0[2]), "+f"(C0[3])
                : "r"(smem[0]), "r"(smem[1]),
                  "r"(smem[2])
            );
            asm (
                "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5 },         "
                "   { %6 },             "
                "   { %0, %1, %2, %3 }; "
                : "+f"(C1[0]), "+f"(C1[1]), "+f"(C1[2]), "+f"(C1[3])
                : "r"(smem[0]), "r"(smem[1]),
                  "r"(smem[2])
            );
        }
    }
    if (C0[0] > 0.2) {
        tensor_out =
            C0[0] + C0[1] + C0[2] + C0[3] +
            C1[0] + C1[1] + C1[2] + C1[3];
    }
#endif // (__CUDA_ARCH__ >= 800)
}

static __global__ void tensor_tf32tf32f32_hammer_kernel()
{
#if (__CUDA_ARCH__ >= 800)
    __shared__ unsigned int smem[6];
    float *A = (float *)&smem[0];
    A[0] = nvcuda::wmma::__float_to_tf32(0.01);
    A[1] = nvcuda::wmma::__float_to_tf32(0.01);
    A[2] = nvcuda::wmma::__float_to_tf32(0.01);
    A[3] = nvcuda::wmma::__float_to_tf32(0.01);

    float *B = (float *)&smem[4];
    B[0] = nvcuda::wmma::__float_to_tf32(0.01);
    B[1] = nvcuda::wmma::__float_to_tf32(0.01);

    float C0[4] = { 0.01, 0.01, 0.01, 0.01 };
    float C1[4] = { 0.01, 0.01, 0.01, 0.01 };
    float C2[4] = { 0.01, 0.01, 0.01, 0.01 };

    __syncthreads();

    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm (
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+f"(C0[0]), "+f"(C0[1]), "+f"(C0[2]), "+f"(C0[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
            asm (
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+f"(C1[0]), "+f"(C1[1]), "+f"(C1[2]), "+f"(C1[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
            asm (
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+f"(C2[0]), "+f"(C2[1]), "+f"(C2[2]), "+f"(C2[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
        }
    }
    if (C0[0] > 0.2) {
        tensor_out =
            C0[0] + C0[1] + C0[2] + C0[3] +
            C1[0] + C1[1] + C1[2] + C1[3] +
            C2[0] + C2[1] + C2[2] + C2[3];
    }
#endif // (__CUDA_ARCH__ >= 800)
}

static __global__ void tensor_f64f64f64_hammer_kernel()
{
#if (__CUDA_ARCH__ >= 800)
    __shared__ unsigned int smem[4];
    double *A = (double *)&smem[0];
    A[0] = 0.01;

    double *B = (double *)&smem[2];
    B[0] = 0.01;

    double C0[2] = { 0.01, 0.01 };
    double C1[2] = { 0.01, 0.01 };

    __syncthreads();

    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm (
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                "   { %0, %1 }, "
                "   { %2 },     "
                "   { %3 },     "
                "   { %0, %1 }; "
                : "+d"(C0[0]), "+d"(C0[1])
                : "d"(*A),
                  "d"(*B)
            );
            asm (
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                "   { %0, %1 }, "
                "   { %2 },     "
                "   { %3 },     "
                "   { %0, %1 }; "
                : "+d"(C1[0]), "+d"(C1[1])
                : "d"(*A),
                  "d"(*B)
            );
        }
    }
    if (C0[0] > 0.2) {
        tensor_out = C0[0] + C0[1] + C1[0] + C1[1];
    }
#endif // (__CUDA_ARCH__ >= 800)
}

static __global__ void tensor_s8s8s32_hammer_kernel()
{
#if (__CUDA_ARCH__ >= 800)
    __shared__ unsigned int smem[6];
    char *A = (char *)&smem[0];
    A[0] = 1;
    A[1] = -1;
    A[2] = 1;
    A[3] = -1;
    A[4] = 1;
    A[5] = -1;
    A[6] = 1;
    A[7] = -1;
    A[8] = 1;
    A[9] = -1;
    A[10] = 1;
    A[11] = -1;
    A[12] = 1;
    A[13] = -1;
    A[14] = 1;
    A[15] = -1;

    char *B = (char *)&smem[4];
    B[0] = 1;
    B[1] = 1;
    B[2] = 1;
    B[3] = 1;
    B[4] = 1;
    B[5] = 1;
    B[6] = 1;
    B[7] = 1;

    int C0[4] = { 1, 1, 1, 1 };
    int C1[4] = { 1, 1, 1, 1 };
    int C2[4] = { 1, 1, 1, 1 };

    __syncthreads();

    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm (
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+r"(C0[0]), "+r"(C0[1]), "+r"(C0[2]), "+r"(C0[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
            asm (
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+r"(C1[0]), "+r"(C1[1]), "+r"(C1[2]), "+r"(C1[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
            asm (
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+r"(C2[0]), "+r"(C2[1]), "+r"(C2[2]), "+r"(C2[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
        }
    }
    if (C0[0] < 0.2) {
        tensor_out =
            C0[0] + C0[1] + C0[2] + C0[3] +
            C1[0] + C1[1] + C1[2] + C1[3] +
            C2[0] + C2[1] + C2[2] + C2[3];
    }
#endif // (__CUDA_ARCH__ >= 800)
}

static __global__ void tensor_s4s4s32_hammer_kernel()
{
#if (__CUDA_ARCH__ >= 800)
    __shared__ unsigned int smem[6];
    unsigned int *A = (unsigned int *)&smem[0];
    A[0] = 0xf1f1f1f1;
    A[1] = 0xf1f1f1f1;
    A[2] = 0xf1f1f1f1;
    A[3] = 0xf1f1f1f1;

    unsigned int *B = (unsigned int *)&smem[4];
    B[0] = 0x11111111;
    B[1] = 0x11111111;

    int C0[4] = { 1, 1, 1, 1 };
    int C1[4] = { 1, 1, 1, 1 };
    int C2[4] = { 1, 1, 1, 1 };

    __syncthreads();

    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm (
                "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+r"(C0[0]), "+r"(C0[1]), "+r"(C0[2]), "+r"(C0[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
            asm (
                "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+r"(C1[0]), "+r"(C1[1]), "+r"(C1[2]), "+r"(C1[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
            asm (
                "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+r"(C2[0]), "+r"(C2[1]), "+r"(C2[2]), "+r"(C2[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
        }
    }
    if (C0[0] < 0.2) {
        tensor_out =
            C0[0] + C0[1] + C0[2] + C0[3] +
            C1[0] + C1[1] + C1[2] + C1[3] +
            C2[0] + C2[1] + C2[2] + C2[3];
    }
#endif // (__CUDA_ARCH__ >= 800)
}

static __global__ void tensor_b1b1s32_hammer_kernel()
{
#if (__CUDA_ARCH__ >= 800)
    __shared__ unsigned int smem[6];
    unsigned int *A = (unsigned int *)&smem[0];
    A[0] = 0xaaaaaaaa;
    A[1] = 0xaaaaaaaa;
    A[2] = 0xaaaaaaaa;
    A[3] = 0xaaaaaaaa;

    unsigned int *B = (unsigned int *)&smem[4];
    B[0] = 0xffffffff;
    B[1] = 0xffffffff;

    int C0[4] = { 1, 1, 1, 1 };
    int C1[4] = { 1, 1, 1, 1 };
    int C2[4] = { 1, 1, 1, 1 };

    __syncthreads();

    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm (
                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+r"(C0[0]), "+r"(C0[1]), "+r"(C0[2]), "+r"(C0[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
            asm (
                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+r"(C1[0]), "+r"(C1[1]), "+r"(C1[2]), "+r"(C1[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
            asm (
                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5, %6, %7 }, "
                "   { %8, %9 },         "
                "   { %0, %1, %2, %3 }; "
                : "+r"(C2[0]), "+r"(C2[1]), "+r"(C2[2]), "+r"(C2[3])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5])
            );
        }
    }
    if (C0[0] < 0.2) {
        tensor_out =
            C0[0] + C0[1] + C0[2] + C0[3] +
            C1[0] + C1[1] + C1[2] + C1[3] +
            C2[0] + C2[1] + C2[2] + C2[3];
    }
#endif // (__CUDA_ARCH__ >= 800)
}

extern "C" {

cudaError_t tensor_f16f16f32_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)tensor_f16f16f32_hammer_kernel, grid, block, 0, 0, s);
}

cudaError_t tensor_f16f16f16_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)tensor_f16f16f16_hammer_kernel, grid, block, 0, 0, s);
}

cudaError_t tensor_bf16bf16f32_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)tensor_bf16bf16f32_hammer_kernel, grid, block, 0, 0, s);
}

cudaError_t tensor_tf32tf32f32_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)tensor_tf32tf32f32_hammer_kernel, grid, block, 0, 0, s);
}

cudaError_t tensor_f64f64f64_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)tensor_f64f64f64_hammer_kernel, grid, block, 0, 0, s);
}

cudaError_t tensor_s8s8s32_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)tensor_s8s8s32_hammer_kernel, grid, block, 0, 0, s);
}

cudaError_t tensor_s4s4s32_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)tensor_s4s4s32_hammer_kernel, grid, block, 0, 0, s);
}

cudaError_t tensor_b1b1s32_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(BLOCK_SIZE, 1, 1);
    return cudaLaunchKernel((void *)tensor_b1b1s32_hammer_kernel, grid, block, 0, 0, s);
}

} // extern "C"
