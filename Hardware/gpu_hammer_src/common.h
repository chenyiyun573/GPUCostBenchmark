#ifndef COMMON_H_
#define COMMON_H_

#define CUDACHECK(cmd) do {                             \
    cudaError_t err = cmd;                              \
    if (err != cudaSuccess) {                           \
        printf("CUDA failure %s:%d '%s'\n",             \
            __FILE__,__LINE__,cudaGetErrorString(err)); \
        exit(1);                                        \
    }                                                   \
} while(0)


#endif // COMMON_H_
