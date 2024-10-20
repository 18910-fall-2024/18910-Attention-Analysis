import os
import subprocess
import torch
from torch.utils.cpp_extension import include_paths, library_paths, CUDA_HOME
import sysconfig
import shutil

def process_user_input(step):
    if step == 'y' or 'Y':
        pass
    elif step == 'n' or 'N':
        print(f"Stopping the program...")
        exit()
    else:
        print(f"Please input \"y\" or \"n\"")


source_dir = "cuda_src"

sources = []
for root, dirs, files in os.walk(source_dir):
    for file in files:
        sources.append(os.path.join(root, file))


python_include_path = sysconfig.get_paths()["include"]

include_dirs = [
    "../../flash-attention/csrc/flash_attn",
    "../../flash-attention/csrc/flash_attn/src",
    "../../flash-attention/csrc/cutlass/include",
    python_include_path,
    *include_paths(),
    f"{CUDA_HOME}/include"
]

include_args = [f"-I{dir}" for dir in include_dirs]

# Generate LLVM IR files for each source code
step_1 = input("Generate LLVM IR file for each source code? [y/n]")
process_user_input(step_1)

ir_dir = "llvm_ir"
if not os.path.exists(ir_dir):
    os.makedirs(ir_dir)

for src in sources:
    if src.endswith(".cpp"):
        print(f"Preprocessing {src}")
        try:
            output_file = os.path.join(ir_dir, os.path.basename(src).replace('.cpp', '.ll'))
            subprocess.run(['clang++', '-O3', '-std=c++17', '-emit-llvm', '-S'] + include_args + [src, '-o', f'{output_file}'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error preprocessing {src}: {e}")
        pass
    
    elif src.endswith(".cu"):
        print(f"Preprocessing {src}")
        try:
            output_file = os.path.basename(src).replace('.cu', '.ll')
            output_path = os.path.join(ir_dir, os.path.basename(src).replace('.cu', '.ll'))
            subprocess.run(['clang++', src,
                            '-O3', '-std=c++17', '-ferror-limit=0',
                            '-w',
                            '-U__CUDA_NO_HALF_OPERATORS__',
                            '-U__CUDA_NO_HALF_CONVERSIONS__',
                            '-U__CUDA_NO_HALF2_OPERATORS__',
                            '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                            '--no-cuda-version-check',
                            '-emit-llvm', '-S', '--cuda-gpu-arch=sm_80'
                            ] + include_args, check=True)
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(output_file, ir_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error preprocessing {src}: {e}")


# Link all LLVM IRs
step_2 = input("Link all LLVM IR files? [y/n]")
process_user_input(step_2)

ir_file = 'linked_ir.ll'
subprocess.run(['llvm-link', f'{ir_dir}/*', '-o', ir_file, '-S'])
print(f"Linked all LLVM IR")


# Convert LLVM IR file to dot file
step_3 = input("Convert LLVM IR file to dot file? [y/n]")
process_user_input(step_3)

subprocess.run(['opt', '--passes=dot-callgraph', ir_file, '-o', '/dev/null'], check=True)

