

## Nvidia GPUs' Parameters

The following are the parameters of some Nvidia GPUs, we list their parameters here and show the Power Benchmark results for them.

| GPU Model        | CUDA Cores | Tensor Cores | GPU Memory | Memory Bandwidth | Memory Interface | Peak Single-Precision Performance | Peak Half-Precision Performance | Form Factor | Approximate Price (USD)   | Power Cap (Watts) |
|------------------|------------|--------------|------------|------------------|------------------|----------------------------------|---------------------------------|-------------|--------------------------|------------------|
| NVIDIA Tesla T4  | 2,560      | 320          | 16 GB      | 320 GB/s         | 256-bit          | 8.1 TFLOPS                      | 65 TFLOPS                       | PCIe        | $2,499                   | 70W              |
| NVIDIA Tesla P40 | 3,840      | N/A          | 24 GB      | 346 GB/s         | 384-bit          | 12.0 TFLOPS                     | 47.5 TFLOPS                     | PCIe        | $7,999                   | 250W             |
| NVIDIA Tesla V100 | 5,120      | 640          | 16 GB or 32 GB | 900 GB/s (16 GB), 1,000 GB/s (32 GB) | 4096-bit | 7.8 TFLOPS (16 GB), 7.5 TFLOPS (32 GB) | 15.7 TFLOPS (16 GB), 15.0 TFLOPS (32 GB) | PCIe or SXM2 | $9,599 (16 GB), $11,599 (32 GB) | 250W (16 GB), 300W (32 GB) |
| NVIDIA Ampere A100 | 6,912    | 432          | 40 GB or 80 GB | 1,555 GB/s | 5120-bit | 9.7 TFLOPS (40 GB), 19.5 TFLOPS (80 GB) | 19.5 TFLOPS (40 GB), 39.0 TFLOPS (80 GB) | PCIe or SXM4 | $11,999 (40 GB), $19,999 (80 GB) | 400W (40 GB), 400W (80 GB) |


## Idle Power

(results on 20230523 by experiments on 8 A100 GPUs on the same node: )
NVIDIA Ampere A100 40GB
We found that different GPUs have slight different mean idle power, from 53.57 Watt to 60.31 Watt.
The idle power consumption of each GPU fluctuates very little, and the maximum std is only 0.16 Watt. 
The idle power range for one GPU is up to 0.5W

Idle Power of A100 40 GB: 53.57 Watt ~ 60.31 Watt +- 0.2 Watt

