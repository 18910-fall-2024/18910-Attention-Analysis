# Flash-Attention-2 Code Analysis

The repo is code for analysis of FlashAttention-2 code base: https://github.com/Dao-AILab/flash-attention


## Dependencies

The repo is developed under Python 3.9.19

Install dependencies by running:

```sh {"id":"01J91W36QY9G7ZVW4BSSQ5Y7XB"}
pip install -r requirements.txt
```

**Recommend**: Create a new environment with python=3.9.19 using Anaconda, and run `pip install -r requirements.txt` in the new environment directly.


## Model Parameters
### Relationship Mapping
To generate a relationship mapping file, run:

```sh
cd relationship_mapping
python getcallgraph.py
```

The program will ask you to input the relative path to the code base to be analyzed and choose the output format.
As we've included the compiled flash-attention code in our code base, you can just type in the path as `../flash-attention`:
```plaintext
Path to the code base: ../flash-attention
```

For the output format, you can choose to generate chain or JSON format. Only output in **chain** format can be used for the parameter identification later.
```plaintext
Output mode: [json/chain]
chain
```

You can choose whether to filter out chains that are not related to CUDA code. (Only chains related to CUDA code can be correctly analyzed in the parameter identification part.)
```plaintext
Only output cuda-related chains? [Y/N]
Y
```

After running the program, you will see a file named `callgraphchain_whole` or `callgraphjson_whole` generated in the current directory.


### Parameter Identification
#### Prerequisite
1. Make sure you have generated the `callgraphchain_whole` output from the relationship mapping part.

2. You need to install and run Ollama server on your machine before running this part.
Follow the instructions here to install Ollama: https://ollama.com/download

**Make sure your Ollama server is running before beginning this part. Do not change the default OLLAMA_HOST and OLLAMA_PORT settings.**

#### Run the code
##### 1. Prompts Generation
Get into the `identification` directory and run `get_prompt.py` to generate prompt for each call chain:
```sh
cd identification
python get_prompt.py
```

The program will ask you to input the code base path as you did in the relationship mapping part. Just input the same path:
```plaintext
Path to the code base: ../flash-attention
```

Then, the program will ask you to input the path to the call chain file you've generated in relationship mapping. Type in `../relationship_mapping/callgraphchain_whole` here:
```plaintext
Path to call graph chain file:../relationship_mapping/callgraphchain_whole
```

After this, you will see a directory called `prompt` generated in current folder, and there are multiple prompt files in the directory.


##### 2. Identification with LLM
Make sure you have run the first step and have `prompt` folder now.
Still in the identification directory, run `param_identification_withapi.py` to start the identification task:
```sh
python param_identification_withapi.py
```
**Note**: Make sure your Ollama server is on when running this script.


After running the command, you'll see prints on your screen showing that the model is working, like:
```plaintext
Analyzing prompt_api_1...
Input token count: 5312
```

Your Ollama server may need to pull the model needed when you run the code for the first time.

After an analysis finish, there will be a print like:
```plaintext
Wrote output to output_prompt_api_1
```

And you can find the output file in `api_output` folder.


## Hardware Parameters
### Relationship Mapping
#### Dependencies
```plaintext
clang+llvm >= 15.0.6
(Using 18.1.8 here)
Install:
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar -xf clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz
sudo mv clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04 /usr/local/llvm-18.1.8 (Or move to anywhere you like)
rm clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz
vim ~/.bashrc
    add:
    export PATH=/usr/local/llvm-18.1.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/llvm-18.1.8/lib:$LD_LIBRARY_PATH
source ~/.bashrc


graphviz
graphviz-dev
```