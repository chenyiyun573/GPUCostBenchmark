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

def output_configuration():
    with open('GPUinfo.md', 'w') as file:
        subprocess.run('nvidia-smi', stdout=file, shell=True, text=True)

# Functions for different parts
def pre_execute():
    print("PreExecute: Start measuring all GPUs' idle power and related stats. Please wait about 6 mins to finish it.")
    # Running a Python script in the './PreExecute/' folder
    pre_execute_output = execute_python_script("./PreExecute/idle.py")
    print("PreExecute done.")

def hardware_execute():
    print('Hardware: Run GPU hardware stress test using compiled .cu scripts. Please wait about 24 mins to finish it.')
    # Running the compiled CUDA program in the './Hardware/' folder
    output = execute_cuda_program("./Hardware/main_loop")
    print('Hardware Done.')

def application_execute():
    print('Application: Train or infer using AI models in pytorch scripts. Please wait about X mins to finish it.')
    execute_models()
    print('Application Done.')

# Main script
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    
    selected_parts = ["PreExecute","Hardware","Application"]
    

    if "PreExecute" in selected_parts:
        pre_execute()
    if "Hardware" in selected_parts:
        hardware_execute()
    if "Application" in selected_parts:
        application_execute()







