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

def convert_ir_to_dot(ir_file, ir_dir, output_dir):
    try:
        ir_file = os.path.join(ir_dir, ir_file)
        output_file = os.path.basename(ir_file) + '.callgraph.dot'
        output_path = os.path.join(dot_folder, output_file)
        output_file = os.path.join(ir_dir, output_file)
        subprocess.run(['opt', '--passes=dot-callgraph', ir_file, '-o', '/dev/null'], check=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.move(output_file, output_dir)
        print(f"Converted {ir_file} to callgraph dot")
        return os.path.basename(output_file)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {ir_file}: {e}")
        exit()


def convert_dot_to_svg(dot_file):
    output_svg = os.path.splitext(dot_file)[0].split('.')[0] + ".svg"
    
    try:
        subprocess.run(['dot', '-Tsvg', dot_file, '-o', output_svg], check=True)
        print(f"Converted {dot_file} to {output_svg}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {dot_file}: {e}")


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


def parse_dot_file(dot_file, parsed_file, dot_folder):
    dot_file = os.path.join(dot_folder, dot_file)
    parsed_file = os.path.join(dot_folder, parsed_file)
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

def convert_dot_to_json(dot_file, output_dir):
    graph = pgv.AGraph(dot_file)
    call_graph = {}

    for node in graph.nodes():
        function = graph.get_node(node).attr['label'].strip("{}")
        function = re_sanitize_label(function)
        call_graph[function] = []

    for edge in graph.edges():
        caller = graph.get_node(edge[0]).attr['label'].strip("{}")
        caller = re_sanitize_label(caller)
        callee = graph.get_node(edge[1]).attr['label'].strip("{}")
        callee = re_sanitize_label(callee)

        call_graph[caller].append(callee)

    json_output = json.dumps(call_graph, indent=4)
    dot_file = os.path.basename(dot_file)
    output_file = os.path.splitext(dot_file)[0].split('.')[0] + ".json"
    output_file = os.path.join(output_dir, output_file)
    with open(output_file, 'w') as f:
        f.write(json_output)
        print(f"Converted {dot_file} to {output_file}")

    return call_graph

if __name__ == "__main__":
    ir_folder = "llvm_ir"
    dot_folder = "callgraph_dot"
    if not os.path.exists(dot_folder):
        os.makedirs(dot_folder)
    json_folder = "callgraph_json"
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    for root, dirs, files in os.walk(ir_folder):
        for ir_file in files:
            print(f'{ir_file}\n')
            dot_file = convert_ir_to_dot(ir_file, ir_folder, dot_folder)

            parsed_dot_file = f'parsed_{dot_file}'
            parse_dot_file(dot_file, parsed_dot_file, dot_folder)

            # convert_dot_to_svg(os.path.join(dot_folder, parsed_dot_file))
    
            parsed_dot_file = os.path.join(dot_folder, parsed_dot_file)
            convert_dot_to_json(parsed_dot_file, json_folder)
