import subprocess

# # Default generation parameters
# '''
# Mirostat Sampling: using perplexity to decide the sampling probability for the next token.
# Mirostat algorithm will dynamically adjust the probability distribution for sampling the 
# next token during generation. Higher perplexity means higher diversity.
# Higher mirostat_eta will make the changing of probability distribution more sensitive to 
# the generated token.
# Higher mirostat_tau will result in higher perplexity.
# '''
# mirostat = 0         # Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
# mirostat_eta = 0.1   # Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive.
# mirostat_tau = 5.0   # Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text.

# num_ctx = 2048       # Sets the size of the context window used to generate the next token. 
# repeat_last_n = 64   # Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
# repeat_penalty = 1.1 # Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.

# temperature = 0.8    # The temperature of the model. Increasing the temperature will make the model answer more creatively
# seed = 0             # Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. 
# tfs_z = 1.0          # Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. 
# num_predict = 128    # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
# top_k = 40           # Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative.
# top_p = 0.9          # Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.
# min_p = 0.0          # Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out.



params = [f"--num_predict -1 --repeat_last_n 128 --repeat_penalty 1.2 --top_k 40 --top_p 0.8 --min_p 0.4 --temperature 0 --tfs_z 1 --num_keep 5 --seed -1"]

for i in range(10):
    param = params[0]
    print(f"Running param_identification_withapi.py with {param}")
    
    subprocess.run(f"python param_identification_withapi.py {param}", shell=True)

    
print("All runs completed.")