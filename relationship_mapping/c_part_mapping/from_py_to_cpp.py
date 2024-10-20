import ast
import json
import os
import glob
import re
import shutil


class Connector:
    def __init__(self, code_base, module_name, setup_file) -> None:
        self.code_base = os.path.abspath(code_base)
        self.setup_file = os.path.abspath(setup_file)
        self.module_name = module_name
        
    def get_source_files(self):
        with open(self.setup_file, "r") as f:
            setup_code = f.read()

        copy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_src")
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir)

        tree = ast.parse(setup_code)
        possible_apis = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "CUDAExtension":
                    for keyword in node.keywords:
                        if keyword.arg == "name":
                            if isinstance(keyword.value, ast.Constant) and keyword.value.s != self.module_name:
                                break
                        if keyword.arg == "sources":
                            if isinstance(keyword.value, ast.List):
                                for source in keyword.value.elts:
                                    src_path = os.path.join(self.code_base, source.s)
                                    dest_path = os.path.join(copy_dir, source.s)
                                    dest_dir = os.path.dirname(dest_path)
                                    if not os.path.exists(dest_dir):
                                        os.makedirs(dest_dir)
                                    shutil.copy(src_path, dest_path)
                                    if isinstance(source, ast.Constant) and source.s.endswith(".cpp"):
                                        possible_apis.append(source.s)
                            
                        elif isinstance(keyword.value, ast.Name):
                                possible_apis.extend(self.find_variable_sources(tree, keyword.value.id))

        return self.find_api_in_possible_apis(possible_apis)
    

    def find_variable_sources(self, tree, var_name):
        sources = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        if isinstance(node.value, ast.List):
                            for source in node.value.elts:
                                if isinstance(source, ast.Constant) and source.s.endswith(".cpp"):
                                    sources.append(source.s)
                        elif isinstance(node.value, ast.Call):
                            # When the source is generated through glob.glob
                            sources.extend(self.evaluate_glob_call(node.value))
        return sources
    
    def evaluate_glob_call(self, call_node):
        sources = []
        if isinstance(call_node.func, ast.Attribute):
            if isinstance(call_node.func.value, ast.Name) and call_node.func.attr == 'glob':
                for arg in call_node.args:
                    if isinstance(arg, ast.Constant):
                        pattern = arg.value
                        sources.extend(glob.glob(pattern))
        return sources

    def find_api_in_possible_apis(self, sources):
        pybind_file = None
        pybind_macro = "PYBIND11_MODULE"
        
        for source in sources:
            source = os.path.abspath(os.path.join(self.code_base, source))
            if not os.path.exists(source):
                continue

            with open(source, "r") as f:
                content = f.read()

                if pybind_macro in content:
                    pybind_file = source
                    break

        return pybind_file


    def pymethod_to_cfunc(self):
        '''
        Only compatible with cases that bind the python methods using PYBIND11_MODULE macro.
        Only have code for the methods binded by def() now.
        '''
        source_file = self.get_source_files()
        code = ""
        with open(source_file, "r") as f:
            code = f.read()

        module_pattern = re.compile(r'PYBIND11_MODULE\((\w+),\s*(\w+)\)\s*{')

        def_pattern = re.compile(r'(\w+)\.def\("(\w+)",\s*&(\w+),\s*"([^"]*)"\);')

        module_match = module_pattern.search(code)
        if not module_match:
            print("No PYBIND11_MODULE found.")
            return {}
        
        module_name = module_match.group(1)
        var_name = module_match.group(2)
        
        mappings = {}
        for match in def_pattern.finditer(code):
            obj_name, py_func, cpp_func, doc = match.groups()
            if obj_name == var_name:
                mappings[py_func] = cpp_func


        return mappings



def find_file_in_codebase(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    
    print(f"Error: '{filename}' not found in codebase '{directory}'")
    return None


if __name__ == "__main__":
    code_base = "../../flash-attention"
    module_name = "flash_attn_2_cuda"
    setup_file = "setup.py"

    setup_file = find_file_in_codebase(code_base, setup_file)
    connector = Connector(code_base=code_base, module_name=module_name, setup_file=setup_file)

    mappings = connector.pymethod_to_cfunc()

    output_file = 'py_to_cpp.json'
    json_output = json.dumps(mappings, indent=4)
    with open(output_file, 'w') as f:
        f.write(json_output)

    print(f"{mappings}")