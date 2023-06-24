nvcc main_one.cu kernel.cu -o main -lnvidia-ml --expt-extended-lambda --gpu-architecture=sm_80


nvcc main_one.cu kernel.cu -o main -lnvidia-ml --expt-extended-lambda --gpu-architecture=sm_70
