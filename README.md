# GPU Power Benchmark

## Introduction
Get the GPU power cost of AI workload effciently.

Folders here:
 - `ApplicationBased` will give results for AI workload like CNN, GPT, and etc.
 - `HardwareBased` will give results based on components of GPU like GPU memory, shared memory, l1 cache, l2 cache, and cores.
 - `RelationExperiments` lists some experiments for relations between workload, hardware and power cost. 
 - `Results` are our GPU power benchmarks' results on different GPUs.

## Disclaimer - Boundary of the Benchmark
We ignore the following factors that affect power consumption of GPUs:

1. Power Supply and distribution in the datacenter


2. Environmental Factors (in the space of the machine):
     - Environmental Temperature
     - Humidity
     - Air flow
     - Air pressure (altitude)


## Getting Started
1. `pip install -r requirements.txt`



