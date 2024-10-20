import os
import subprocess
import pygraphviz as pgv
import re
import json
import shutil

def sanitize_label(label):
    '''
    Sanitize some special characters in parsed labels in case of error
    '''
    return label.replace('{', '&#123;').replace('}', '&#125;').replace('<', '&lt;').replace('>', '&gt;')

def re_sanitize_label(label):
    '''
    Reverse sanitization for json output
    '''
    return label.replace('&#123;', '{').replace('&#125;', '}').replace('&lt;', '<').replace('&gt;', '>')

def reverse_func_name(encoded_name):
    '''
    Reverse function names in the dot file to original names
    '''
    try:
        result = subprocess.run(['c++filt', encoded_name], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error reversing {encoded_name}: {e}")
        return encoded_name


def parse_dot_file(dot_file):
    parsed_file = f'parsed_{dot_file}'
    with open(dot_file, 'r') as infile, open(parsed_file, 'w') as outfile:
        for line in infile:
            match = re.search(r'\s*Node[^\s]+ \[shape=record,label="\{([^}]+)\}"\];', line)
            if match:
                # If the line is a node
                encoded_name = match.group(1)
                original_name = reverse_func_name(encoded_name)
                sanitized_name = sanitize_label(original_name)
                
                new_label = f'{{{sanitized_name}}}'
                # print(new_label)
                new_line = re.sub(r'label="\{[^}]+\}"', f'label="{new_label}"', line)
                outfile.write(new_line)
            else:
                # If the line is not a node
                outfile.write(line)

    return parsed_file

def convert_dot_to_json(dot_file):
    graph = pgv.AGraph(dot_file)
    call_graph = {}

    for node in graph.nodes():
        function = graph.get_node(node).attr['label'].strip("{}")
        function = re_sanitize_label(function)
        if not function.startswith('llvm.'):
            call_graph[function] = []

    for edge in graph.edges():
        caller = graph.get_node(edge[0]).attr['label'].strip("{}")
        caller = re_sanitize_label(caller)
        callee = graph.get_node(edge[1]).attr['label'].strip("{}")
        callee = re_sanitize_label(callee)
        if not caller.startswith('llvm.') and not callee.startswith('llvm.'):
            call_graph[caller].append(callee)

    json_output = json.dumps(call_graph, indent=4)
    dot_file = os.path.basename(dot_file)
    output_file = os.path.splitext(dot_file)[0].split('.')[0] + ".json"
    with open(output_file, 'w') as f:
        f.write(json_output)
        print(f"Converted {dot_file} to {output_file}")

    return call_graph


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

def get_starting_functions(mapping_file):
    mappings = {}
    with open(mapping_file, 'r') as f:
        mappings = json.load(f)

    starting_funcs = set()
    for _, value in mappings.items():
        starting_funcs.add(value)
        starting_funcs.add(f'void {value}')
        starting_funcs.add(f'void* {value}')

    return starting_funcs


def get_call_chains(function_calls, start_funcs):
    call_chains = set()

    def dfs(func, chain, visited):
        if func in visited:
            # Detected a cycle
            # call_chains.add(" -> ".join(chain + [func]) + " (cycle)")
            return
        
        if len(chain) == 0:
            if not func.startswith(tuple(start_funcs)):
                return
            
        visited.add(func)
        if len(chain) != 0 and func.startswith(chain[-1]):
            return
        chain.append(func)

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
    dot_file = 'linked_ir.ll.callgraph.dot'
    map_file = '../py_to_cpp.json'
    parsed_dot_file = parse_dot_file(dot_file)
    call_graph = convert_dot_to_json(parsed_dot_file)

    start_functions = get_starting_functions(map_file)    
    call_chains = get_call_chains(call_graph, start_functions)
    output_file = 'callgraphchain'
    with open(output_file, 'w', encoding='utf-8') as f:
        for chain in call_chains:
            f.write(chain + '\n')

