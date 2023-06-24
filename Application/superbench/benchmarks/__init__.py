# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exposes interfaces of benchmarks used by SuperBench executor."""

import importlib

from Application.superbench.benchmarks.return_code import ReturnCode
from Application.superbench.benchmarks.context import Platform, Framework, Precision, ModelAction, \
    DistributedImpl, DistributedBackend, BenchmarkType, BenchmarkContext
from Application.superbench.benchmarks.reducer import ReduceType, Reducer
from Application.superbench.common.utils import LazyImport

BenchmarkRegistry = LazyImport(
    'Application.superbench.benchmarks.registry', 'BenchmarkRegistry', lambda: list(
        map(
            importlib.import_module, [
                'Application.superbench.benchmarks.{}'.format(module)
                for module in ['model_benchmarks']
            ]
        )
    )
)

__all__ = [
    'ReturnCode', 'Platform', 'Framework', 'BenchmarkType', 'Precision', 'ModelAction', 'DistributedImpl',
    'DistributedBackend', 'BenchmarkContext', 'BenchmarkRegistry', 'ReduceType', 'Reducer'
]
