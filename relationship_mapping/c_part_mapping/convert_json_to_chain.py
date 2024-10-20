import os
import json
from collections import defaultdict

def merge_dicts(main_dict, new_dict):
    for key, value in new_dict.items():
        value = set(value)
        if key in main_dict:
            main_dict[key].update(value)
        else:
            main_dict[key] = value


def read_json_files(folder_path):
    merged_data = defaultdict(set)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r') as json_file:
            file_data = json.load(json_file)
            merge_dicts(merged_data, file_data)

    return merged_data


def find_outermost_functions(function_calls):
    """
    function_calls: a map from functions to functions called by them

    Return a map only including keys that are outermost functions
    """
    called_functions = set()

    for caller, callees in function_calls.items():
        for callee in callees:
            called_functions.add(callee)
    
    outermost_functions = [func for func in function_calls if func not in called_functions]
    
    return outermost_functions


def get_call_chains(function_calls):
    call_chains = set()

    def dfs(func, chain, visited):
        if func in visited:
            # Detected a cycle
            # call_chains.add(" -> ".join(chain + [func]) + " (cycle)")
            return
        
        chain.append(func)
        visited.add(func)

        if func in function_calls and function_calls[func]:
            for called_func in function_calls[func]:
                dfs(called_func, chain[:], visited)
        else:
            call_chains.add(" -> ".join(chain))
        
        visited.remove(func)

    
    outermost_functions = find_outermost_functions(function_calls)
    for func in outermost_functions:
        dfs(func, [], set())

    return sorted(list(call_chains))

if __name__ == "__main__":
    json_folder = 'callgraph_json'
    function_calls = read_json_files(json_folder)
    call_chains = get_call_chains(function_calls)

    output_file = 'callgraphchain'
    with open(output_file, 'w', encoding='utf-8') as f:
        for chain in call_chains:
            f.write(chain + '\n')

