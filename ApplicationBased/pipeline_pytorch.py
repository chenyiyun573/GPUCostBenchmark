# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from superbench.benchmarks import Platform, Framework, BenchmarkRegistry
from superbench.common.utils import logger

def run_benchmark(model_name, parameters):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--distributed', action='store_true', default=False, help='Whether to enable distributed training.'
    )
    args = parser.parse_args()

    # add distributed to parameters if distributed flag is passed
    if args.distributed:
        parameters += ' --distributed_impl ddp --distributed_backend nccl'

    # Create context for benchmark and run it.
    context = BenchmarkRegistry.create_benchmark_context(
        model_name, platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )

if __name__ == '__main__':
    # bert-large benchmark
    model_name = 'bert-large'
    parameters = '--batch_size 1 --duration 120 --seq_len 128 --precision float32 --run_count 2 --model_action inference'
    run_benchmark(model_name, parameters)

    # resnet101 benchmark
    model_name = 'resnet101'
    parameters = '--batch_size 192 --precision float32 float16 --num_warmup 64 --num_steps 512 --sample_count 8192 --pin_memory'
    run_benchmark(model_name, parameters)

    # gpt2-large benchmark
    model_name = 'gpt2-large'
    parameters = '--batch_size 1 --duration 120 --seq_len 128 --precision float32 --run_count 2'
    run_benchmark(model_name, parameters)

    # lstm benchmark
    model_name = 'lstm'
    parameters = '--batch_size 1 --seq_len 256 --precision float32 --num_warmup 8 --num_steps 64 --run_count 2'
    run_benchmark(model_name, parameters)
