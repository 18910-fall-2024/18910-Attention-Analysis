import argparse
import os
from datetime import datetime
import ollama
from transformers import AutoTokenizer
import torch

ollama.pull('qwen2.5:14b')

parser = argparse.ArgumentParser()
'''
Mirostat Sampling: using perplexity to decide the sampling probability for the next token.
Mirostat algorithm will dynamically adjust the probability distribution for sampling the 
next token during generation. Higher perplexity means higher diversity.
Higher mirostat_eta will make the changing of probability distribution more sensitive to 
the generated token.
Higher mirostat_tau will result in higher perplexity.
'''
parser.add_argument('--mirostat', type=int, default=0, help="Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)")
parser.add_argument('--mirostat_eta', type=float, default=0.1, help="Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)")
parser.add_argument('--mirostat_tau', type=float, default=5.0, help="Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. (Default: 5.0)")
parser.add_argument('--repeat_last_n', type=int, default=128, help="Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)")
parser.add_argument('--repeat_penalty', type=float, default=1.2, help="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)")
parser.add_argument('--temperature', type=float, default=0.2, help="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)")
parser.add_argument('--seed', type=int, default=-1, help="Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)")
parser.add_argument('--tfs_z', type=float, default=1.0, help="Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)")
parser.add_argument('--num_predict', type=int, default=-1, help="Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)")
parser.add_argument('--top_k', type=int, default=40, help="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)")
parser.add_argument('--top_p', type=float, default=0.8, help="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)")
parser.add_argument('--min_p', type=float, default=0.4, help="Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out. (Default: 0.0)")
parser.add_argument('--num_keep', type=int, default=5, help="Number of context tokens considered for generating the next token")
parser.add_argument('--prompt', type=str, default="prompt_api", help="Prompt file")

args = parser.parse_args()



with open(args.prompt, 'r', encoding='utf-8') as f:
    prompt = f.read()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")
tokens = tokenizer.tokenize(prompt)
token_count = len(tokens)
print(f"Input token count: {token_count}")

num_ctx = token_count + 512

output = f"mirostat={args.mirostat}, mirostat_eta={args.mirostat_eta}, mirostat_tau={args.mirostat_tau}\n"
output += f"num_ctx={num_ctx}, num_keep={args.num_keep}, repeat_last_n={args.repeat_last_n}, repeat_penalty={args.repeat_penalty}\n"
output += f"temperature={args.temperature}, seed={args.seed}, tfs_z={args.tfs_z}, num_predict={args.num_predict}\n"
output += f"top_k={args.top_k}, top_p={args.top_p}, min_p={args.min_p}\n"
output += f"prompt={args.prompt}\n\n"

output += "==================== Generated Output ====================\n"

response = ollama.chat(
    model='qwen2.5:14b',
    messages=[{'role': 'user', 'content': prompt}],
    options={
        'mirostat': args.mirostat,
        'mirostat_eta': args.mirostat_eta,
        'mirostat_tau': args.mirostat_tau,
        'num_ctx': num_ctx,
        'repeat_last_n': args.repeat_last_n,
        'repeat_penalty': args.repeat_penalty,
        'temperature': args.temperature,
        'seed': args.seed,
        'tfs_z': args.tfs_z,
        'num_predict': args.num_predict,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'min_p': args.min_p,
        'num_keep': args.num_keep
    },
    format='json'
)
    
output += f"{response['message']['content']}\n\n"


output_dir = "api_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
now = datetime.now()
formatted_time = now.strftime("%Y%m%d%H%M%S")
with open(os.path.join(output_dir, f"output_{formatted_time}.json"), 'w') as f:
        f.write(output)
        
torch.cuda.empty_cache()