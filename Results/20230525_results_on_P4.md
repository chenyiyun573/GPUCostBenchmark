20230525 1623
One potential error here and in my code, sample/(s*Watt) may not be correct, because per step contains many sample in a batch.

# 20230525 Tencent Cloud one P4 GPU only:

Configurations:
    # Specify the model name and benchmark parameters.
    model_name = 'lstm'
    parameters = '--batch_size 1 --seq_len 256 --precision float32 --num_warmup 8 --num_steps 64 --run_count 2'
    if args.distributed:
        parameters += ' --distributed_impl ddp --distributed_backend nccl'

[2023-05-25 15:24:15,241 VM-0-10-ubuntu:9631][model_base.py:205][INFO] Model placement - model: pytorch-lstm, GPU availablility: True, pin memory: False, force fp32: False.
/home/ubuntu/.local/lib/python3.6/site-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
[2023-05-25 15:25:01,394 VM-0-10-ubuntu:9631][model_base.py:279][INFO] Average train time - round: 0, model: pytorch-lstm, precision: float32, step time: 569.525320 ms, power metric: 0.037029 sample/(s*Watt), power statistic: Mean: 47.41, Max: 49.01, Min: 42.31, STD: 1.45
[2023-05-25 15:25:43,653 VM-0-10-ubuntu:9631][model_base.py:279][INFO] Average train time - round: 1, model: pytorch-lstm, precision: float32, step time: 569.823898 ms, power metric: 0.036119 sample/(s*Watt), power statistic: Mean: 48.60, Max: 50.21, Min: 43.45, STD: 1.50
[2023-05-25 15:25:43,694 VM-0-10-ubuntu:9631][pytorch_lstm.py:39][INFO] benchmark: pytorch-lstm, return code: 0, result: {'return_code': [0], 'fp32_train_step_time': [569.5253200829029, 569.8238983750343], 'fp32_train_throughput': [1.7558556159559908, 1.7549300465825837]}


Configurations:
    # Specify the model name and benchmark parameters.
    model_name = 'bert-large'
    parameters = '--batch_size 1 --duration 120 --seq_len 128 --precision float32 --run_count 2'
    if args.distributed:
        parameters += ' --distributed_impl ddp --distributed_backend nccl'


ubuntu@VM-0-10-ubuntu:~/GPUCostBenchmark/ApplicationBased$ python3 pytorch_bert_large.py
[2023-05-25 15:30:00,493 VM-0-10-ubuntu:11144][model_base.py:205][INFO] Model placement - model: pytorch-bert-large, GPU availablility: True, pin memory: False, force fp32: False.
/home/ubuntu/.local/lib/python3.6/site-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
/home/ubuntu/.local/lib/python3.6/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  FutureWarning,
[2023-05-25 15:32:15,190 VM-0-10-ubuntu:11144][model_base.py:279][INFO] Average train time - round: 0, model: pytorch-bert-large, precision: float32, step time: 310.888148 ms, power metric: 0.064091 sample/(s*Watt), power statistic: Mean: 50.16, Max: 60.86, Min: 40.65, STD: 5.28

[2023-05-25 15:34:26,734 VM-0-10-ubuntu:11144][model_base.py:279][INFO] Average train time - round: 1, model: pytorch-bert-large, precision: float32, step time: 310.766757 ms, power metric: 0.061551 sample/(s*Watt), power statistic: Mean: 52.27, Max: 63.48, Min: 43.35, STD: 5.33
[2023-05-25 15:34:27,033 VM-0-10-ubuntu:11144][pytorch_bert_large.py:39][INFO] benchmark: pytorch-bert-large, return code: 0, result: {'return_code': [0], 'fp32_train_step_time': [310.88814777987346, 310.7667565345764], 'fp32_train_throughput': [3.216592899294065, 3.217848757500554]}



Configurations:(192 batch size run out of memory, change to 32)
    # Specify the model name and benchmark parameters.
    # For example, resnet50, resnet101, resnet152, densenet169, densenet201, vgg11, vgg13, vgg16, vgg19.
    model_name = 'resnet101'
    parameters = '--batch_size 32 --precision float32 float16 --num_warmup 64 --num_steps 512 \
        --sample_count 8192 --pin_memory'

ubuntu@VM-0-10-ubuntu:~/GPUCostBenchmark/ApplicationBased$ python3 pytorch_cnn.py
benchmark launch now
[2023-05-25 15:40:10,275 VM-0-10-ubuntu:13853][model_base.py:205][INFO] Model placement - model: pytorch-resnet101, GPU availablility: True, pin memory: True, force fp32: False.
/home/ubuntu/.local/lib/python3.6/site-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
[2023-05-25 15:46:25,794 VM-0-10-ubuntu:13853][model_base.py:279][INFO] Average train time - round: 0, model: pytorch-resnet101, precision: float32, step time: 615.762825 ms, power metric: 0.026144 sample/(s*Watt), power statistic: Mean: 61.86, Max: 73.79, Min: 28.03, STD: 5.40
[2023-05-25 15:52:00,485 VM-0-10-ubuntu:13853][model_base.py:279][INFO] Average train time - round: 0, model: pytorch-resnet101, precision: float16, step time: 566.895816 ms, power metric: 0.030215 sample/(s*Watt), power statistic: Mean: 58.35, Max: 70.89, Min: 28.23, STD: 6.42
benchmark completed
[2023-05-25 15:52:00,605 VM-0-10-ubuntu:13853][pytorch_cnn.py:45][INFO] benchmark: pytorch-resnet101, return code: 0, result: {'return_code': [0], 'fp32_train_step_time': [615.7628251239657], 'fp32_train_throughput': [51.96883587145319], 'fp16_train_step_time': [566.8958160094917], 'fp16_train_throughput': [56.44827955490836]}


Configurations:
GPT2-large failed because of running out memory. GPT2 Benchmark not implemented.


# The following are results on A100 20230525
LSTM

[2023-05-25 08:14:59,741 a100-dev-000018:65753][model_base.py:205][INFO] Model placement - model: pytorch-lstm, GPU availablility: True, pin memory: False, force fp32: False.
[2023-05-25 08:15:09,862 a100-dev-000018:65753][model_base.py:279][INFO] Average train time - round: 0, model: pytorch-lstm, precision: float32, step time: 82.462981 ms, power metric: 0.049776 sample/(s*Watt), power statistic: Mean: 244.32, Max: 319.11, Min: 202.04, STD: 32.52
[2023-05-25 08:15:16,896 a100-dev-000018:65753][model_base.py:279][INFO] Average train time - round: 1, model: pytorch-lstm, precision: float32, step time: 83.129708 ms, power metric: 0.048345 sample/(s*Watt), power statistic: Mean: 248.81, Max: 319.71, Min: 208.24, STD: 34.37
[2023-05-25 08:15:16,905 a100-dev-000018:65753][pytorch_lstm.py:39][INFO] benchmark: pytorch-lstm, return code: 0, result: {'return_code': [0], 'fp32_train_step_time': [82.46298134326935, 83.12970772385597], 'fp32_train_throughput': [12.12752407493479, 12.030420452937317]}


BERT large

[2023-05-25 08:16:56,540 a100-dev-000018:72075][model_base.py:205][INFO] Model placement - model: pytorch-bert-large, GPU availablility: True, pin memory: False, force fp32: False.
/home/superbench/v-yiyunchen/ytorch-venv/lib/python3.6/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  FutureWarning,
[2023-05-25 08:19:01,549 a100-dev-000018:72075][model_base.py:279][INFO] Average train time - round: 0, model: pytorch-bert-large, precision: float32, step time: 71.528049 ms, power metric: 0.090590 sample/(s*Watt), power statistic: Mean: 154.33, Max: 191.78, Min: 86.59, STD: 19.33
[2023-05-25 08:21:03,342 a100-dev-000018:72075][model_base.py:279][INFO] Average train time - round: 1, model: pytorch-bert-large, precision: float32, step time: 72.188608 ms, power metric: 0.089900 sample/(s*Watt), power statistic: Mean: 154.20, Max: 191.45, Min: 86.25, STD: 19.20
[2023-05-25 08:21:03,413 a100-dev-000018:72075][pytorch_bert_large.py:39][INFO] benchmark: pytorch-bert-large, return code: 0, result: {'return_code': [0], 'fp32_train_step_time': [71.52804934109251, 72.18860788270831], 'fp32_train_throughput': [13.984968876161888, 13.854431350832222]}


    model_name = 'resnet101'
    parameters = '--batch_size 32 --precision float32 float16 --num_warmup 64 --num_steps 512 \
        --sample_count 8192 --pin_memory'

W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[2023-05-31 01:38:58,932 a100-dev-000018:69095][model_base.py:279][INFO] Average train time - round: 0, model: pytorch-resnet101, precision: float32, step time: 79.808518 ms, power metric: 0.043715 sample/(s*Watt), power statistic: Mean: 285.41, Max: 325.99, Min: 90.78, STD: 34.49
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[2023-05-31 01:39:36,414 a100-dev-000018:69095][model_base.py:279][INFO] Average train time - round: 0, model: pytorch-resnet101, precision: float16, step time: 58.778138 ms, power metric: 0.071880 sample/(s*Watt), power statistic: Mean: 236.39, Max: 289.75, Min: 90.50, STD: 33.02
benchmark completed
[2023-05-31 01:39:36,434 a100-dev-000018:69095][pytorch_cnn.py:45][INFO] benchmark: pytorch-resnet101, return code: 0, result: {'return_code': [0], 'fp32_train_step_time': [79.80851782485843], 'fp32_train_throughput': [401.29244804793734], 'fp16_train_step_time': [58.77813836559653], 'fp16_train_throughput': [545.2521443318288]}



Certainly, here is the requested table:

| Model      | GPU  | Power Efficiency (step/(s*Watt)) | Step Time (ms) | Power Statistic (Mean) | Power Statistic (Max) | Power Statistic (Min) | Power Statistic (STD) |
|------------|------|---------------------------------|-----------------|------------------------|-----------------------|-----------------------|-----------------------|
| LSTM       | P4   | 0.037029                        | 569.525320      | 47.41                  | 49.01                 | 42.31                 | 1.45                  |
| LSTM       | A100 | 0.049776                        | 82.462981       | 244.32                 | 319.11                | 202.04                | 32.52                 |
| Bert-large | P4   | 0.064091                        | 310.888148      | 52.27                  | 63.48                 | 43.35                 | 5.33                  |
| Bert-large | A100 | 0.090590                        | 71.528049       | 154.20                 | 191.45                | 86.25                 | 19.20                 |
| Resnet101  | P4   | 0.026144                        | 615.762825      | 61.86                  | 73.79                 | 28.03                 | 5.40                  |
| Resnet101  | A100 | 0.071880                        | 79.808518       | 236.39                 | 289.75                | 90.50                 | 33.02                 |
