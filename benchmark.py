import subprocess
import os

from Application.run_models import run_benchmark

def execute_models():
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


    print('Model Benchmarking completed!')

def execute_cuda_program(file_path):
    process = subprocess.run(file_path, stdout=subprocess.PIPE, universal_newlines=True)
    return process.stdout

def execute_python_script(script_path):
    process = subprocess.run(['python3', script_path], stdout=subprocess.PIPE, universal_newlines=True)
    return process.stdout

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("PreExecute: Start measuring all GPUs' idle power and related statics. Please wait about 6 mins to finish it.")
    # Running a Python script in the './PreExecute/' folder
    pre_execute_output = execute_python_script("./PreExecute/idle.py")
    #print(pre_execute_output)
    print("PreExecute done.")

    print('Hardware: Run GPU hardware stress test using complied .cu scripts. Please wait about X mins to finish it.')
    # Running the compiled CUDA program in the './Hardware/' folder
    output = execute_cuda_program("./Hardware/main_loop")
    #print(output)
    print('Hardware Done.')

    print('Application: Train or infer using AI models in pytorch scripts. Please wait about X mins to finish it.')
    execute_models()
    print('Application Done.')






