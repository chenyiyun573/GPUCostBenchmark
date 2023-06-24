#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"
#include <chrono>
#include "kernels.h"
#include <iostream>
#include <iomanip>

#define CUDACHECK(cmd)                                     \
  do                                                       \
  {                                                        \
    cudaError_t err = cmd;                                 \
    if (err != cudaSuccess)                                \
    {                                                      \
      printf("CUDA failure %s:%d '%s'\n",                  \
             __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                             \
    }                                                      \
  } while (0)

#ifndef _COLORS_
#define _COLORS_

/* FOREGROUND */
#define RST "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x) "\x1B[1m" x RST
#define UNDL(x) "\x1B[4m" x RST

#endif /* _COLORS_ */

int main(int argc, const char *argv[])
{
  std::cout << "\t[" << FRED("GPU Hammer") << "] start" << std::endl;
  auto startTime = std::chrono::system_clock::now().time_since_epoch().count();
  //CUDACHECK(fp_hammer(cudaStreamDefault, 108));
  // CUDACHECK(gmem_fp_hammer(cudaStreamDefault, 108));
  // CUDACHECK(tensor_f16f16f16_hammer(cudaStreamDefault, 108));
  // CUDACHECK(tensor_f16f16f32_hammer(cudaStreamDefault, 108));
  // CUDACHECK(tensor_bf16bf16f32_hammer(cudaStreamDefault, 108));
  // CUDACHECK(tensor_tf32tf32f32_hammer(cudaStreamDefault, 108));
  // CUDACHECK(tensor_f64f64f64_hammer(cudaStreamDefault, 108));
  // CUDACHECK(tensor_s8s8s32_hammer(cudaStreamDefault, 108));
  // CUDACHECK(tensor_s4s4s32_hammer(cudaStreamDefault, 108));
  // CUDACHECK(tensor_b1b1s32_hammer(cudaStreamDefault, 108));
  // CUDACHECK(fp32_hammer(cudaStreamDefault, 108));
  // CUDACHECK(fp64_hammer(cudaStreamDefault, 108));
  // CUDACHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  // CUDACHECK(l1_ld_hammer(cudaStreamDefault, 108));
  // CUDACHECK(l2_ld_hammer(cudaStreamDefault, 108));
  CUDACHECK(gmem_ld_hammer(cudaStreamDefault, 108));
  // CUDACHECK(smem_ld_hammer(cudaStreamDefault, 108));
  CUDACHECK(cudaDeviceSynchronize());
  auto endTime = std::chrono::system_clock::now().time_since_epoch().count();
  std::cout << "\t[" << FRED("GPU Hammer") << "] "
            << "start time = " << std::setprecision(16) << startTime / 1e9 << ", "
            << "end time = " << endTime / 1e9 << ", "
            << "elasped time = " << (endTime - startTime) / 1e9
            << std::endl;
  cudaDeviceReset();
  return 0;
}
