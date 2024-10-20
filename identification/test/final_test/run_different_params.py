import subprocess

params = [f"--temperature 0 --prompt prompt_api_8"]

for i in range(3):
    param = params[0]
    print(f"Running param_identification_withapi.py with {param}")
    
    subprocess.run(f"python param_identification_withapi.py {param}", shell=True)
    
 