# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from Application.superbench.benchmarks import Platform, Framework, BenchmarkRegistry
from Application.superbench.common.utils import logger

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
