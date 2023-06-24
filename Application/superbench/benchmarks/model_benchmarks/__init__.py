# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A module containing all the e2e model related benchmarks."""

from Application.superbench.benchmarks.model_benchmarks.model_base import ModelBenchmark
from Application.superbench.benchmarks.model_benchmarks.pytorch_bert import PytorchBERT
from Application.superbench.benchmarks.model_benchmarks.pytorch_gpt2 import PytorchGPT2
from Application.superbench.benchmarks.model_benchmarks.pytorch_cnn import PytorchCNN
from Application.superbench.benchmarks.model_benchmarks.pytorch_lstm import PytorchLSTM

__all__ = ['ModelBenchmark', 'PytorchBERT', 'PytorchGPT2', 'PytorchCNN', 'PytorchLSTM']
