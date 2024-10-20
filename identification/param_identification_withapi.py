import argparse
import os
import ollama
from transformers import AutoTokenizer
import torch

# ollama.pull('qwen2.5:32b')

prompts_path = "prompt"

output_dir = "api_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
        
for filename in os.listdir(prompts_path):
    print(f"Analyzing {filename}...")
    file_path = os.path.join(prompts_path, filename)

    with open(file_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
        
        
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B")
    tokens = tokenizer.tokenize(prompt)
    token_count = len(tokens)
    print(f"Input token count: {token_count}")
    
    num_ctx = token_count + 512

    response = ollama.chat(
        model='qwen2.5:32b',
        messages=[{'role': 'user', 'content': prompt}],
        options={
            'mirostat': 0,
            'mirostat_eta': 0.1,
            'mirostat_tau': 5,
            'num_ctx': num_ctx,
            'repeat_last_n': 128,
            'repeat_penalty': 1.2,
            'temperature': 0.2,
            'seed': -1,
            'tfs_z': 1.0,
            'num_predict': -1,
            'top_k': 40,
            'top_p': 0.8,
            'min_p': 0.4,
            'num_keep': 5
        },
        format='json'
    )
        
    output = f"{response['message']['content']}\n\n"
        
    output_file = f"output_{filename}.json"
    with open(os.path.join(output_dir, output_file), 'w') as f:
            f.write(output)
    
    print(f"Wrote output to {output_file}\n")
            
    torch.cuda.empty_cache()