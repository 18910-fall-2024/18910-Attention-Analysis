import ast
import clang.cindex
import os
import sys
from collections import defaultdict
import json





def find_outermost_functions(function_calls):
    """
    function_calls: a map from functions to functions called by them

    Return a map only including keys that are outermost functions
    """
    called_functions = set()

    for caller, callees in function_calls.items():
        # caller_source = caller.split("/")[-1]
        for callee in callees:
            callee_source = "/" + callee.split("/")[-1]
            # if callee != caller:
            called_functions.add(callee)
            # if callee_source != caller:
            called_functions.add(callee_source)
    
    outermost_functions = [func for func in function_calls if func not in called_functions]
    
    return outermost_functions


def get_call_chains(function_calls, imports, func_defs, class_defs):
    call_chains = set()  # Use set to avoid repetition
    # print(f"imports\n{imports}")

    def dfs(func, chain, visited):
        if func in visited:
            # Detected a cycle
            # call_chains.add(" -> ".join(chain + [func]) + " (cycle)")
            return
        
        if len(chain) == 1:
            scope = chain[0].split("/")[0]
            source = chain[0].split("/")[-1]
            source_module = source.rsplit('.', 1)[0]
            if (scope == "" or scope in class_defs) and source in func_defs and source_module in class_defs:
                # Filter out the chains starting with a method definition in a class that we don't care about
                return
        
        source_name = func.split("/")[-1]
        # print(f"{source_name}") 
        module_name = source_name.rsplit('.', 1)[0]
        source = "/" + source_name
        if (source_name in imports and source_name not in class_defs) or (source_name in func_defs) or (module_name in imports and module_name not in func_defs and module_name not in class_defs):
            # filter out items that are only variables or class names
            chain.append(func)
        visited.add(func)

        if (func in function_calls and function_calls[func]) or (source in function_calls and function_calls[source]):
            if func in function_calls and function_calls[func]:
                for called_func in function_calls[func]:
                    dfs(called_func, chain[:], visited)

            if source in function_calls and function_calls[source]:
                for called_func in function_calls[source]:
                    dfs(called_func, chain[:], visited)
        else:
            call_chains.add(" -> ".join(chain))
        
        visited.remove(func)

    
    outermost_functions = find_outermost_functions(function_calls)
    for func in outermost_functions:
        dfs(func, [], set())

    return sorted(list(call_chains))



def get_relationship_map(code_base_path, output_json, output_chain):
    '''
    Output relationship mapping files for each file in the code base.
    The structure of output directory would be the same as the code base structure.
    '''
    all_calls = defaultdict(list)
    all_imports = set()
    all_function_defs = set()
    all_class_defs = set()

    module_path = ""
    function_calls = defaultdict(list)
    imports = set()
    func_defs = set()
    class_defs = set()
    for root, _, files in os.walk(code_base_path):
        for file in files:
            file_path = os.path.join(root, file)

            function_calls = defaultdict(list)
            if file.endswith(".py"):
                converter = CodeConverter(code_base_path, file_path)
                module_path, function_calls, imports, func_defs, class_defs = converter.analyze_python_file()
            # elif file.endswith(('.cu', '.cpp', '.h', '.hpp')):
            #     function_calls = self.analyze_c_file(file_path)

            # root_functions.extend(find_root_functions(function_calls))
            # outermost_functions.extend(find_outermost_functions(function_calls))

            # for func, variables in function_variables.items():
            #     all_func_variables[func].extend(variables)

            for func, calls in function_calls.items():
                # function_calls[func] = list(calls)
                all_calls[func].extend(calls)

            all_imports |= imports
            for func_def in func_defs:
                all_function_defs.add(f"{module_path}." + func_def)
            
            all_class_defs |= class_defs

            rel_path = os.path.relpath(file_path, start=code_base_path)
            file_name, file_ext = os.path.splitext(rel_path)
            file_suffix = file_ext[1:]
            
            if output_json:
                output_file = os.path.join('callgraphjson', file_name + '_' + file_suffix)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(function_calls, f, ensure_ascii=False, indent=4)

                # print(f"Function call json saved to {output_file}")

            if output_chain:
                call_chains = get_call_chains(function_calls, imports, func_defs, class_defs)

                output_file = os.path.join('callgraphchain', file_name + '_' + file_suffix)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    for chain in call_chains:
                        f.write(chain + '\n')

                # print(f"Function call chains saved to {output_file}")

    return all_calls, all_imports, all_function_defs, all_class_defs



def get_relationship_map_in_whole(code_base, output_json, output_chain, filter_cuda):
    '''
    Output a relationship mapping file including all function call chains within the code base.
    '''
    all_calls, all_imports, all_function_defs, all_class_defs = get_relationship_map(code_base, False, False)

    if output_json:
        output_file = os.path.join('callgraphjson_whole')

        if filter_cuda:
            filtered_json = {caller: callees for caller, callees in all_calls.items() 
                    if "flash_attn_2_cuda" in caller or 
                        any("flash_attn_2_cuda" in callee for callee in callees)}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_json, f, ensure_ascii=False, indent=4)

        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_calls, f, ensure_ascii=False, indent=4)

        # print(f"Function call json saved to {output_file}")

    if output_chain:
        call_chains = get_call_chains(all_calls, all_imports, all_function_defs, all_class_defs)

        output_file = os.path.join('callgraphchain_whole')

        if filter_cuda:
            filtered_chains = [chain for chain in call_chains if "flash_attn_2_cuda" in chain]
            with open(output_file, 'w', encoding='utf-8') as f:
                for chain in filtered_chains:
                    f.write(chain + '\n\n')

        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                for chain in call_chains:
                    f.write(chain + '\n\n')

        # print(f"Function calls saved to {output_file}")






def get_full_function_path(called_path, source_path):
        return f"{called_path}/{source_path}"




def get_module_path(file_name, code_base_path):
    """
    Generate module path to locate functions in output file
    """
    rel_path = os.path.relpath(file_name, start=code_base_path)
    module_path = rel_path.replace(os.sep, '.').rsplit('.', 1)[0]
    return module_path










class CodeConverter:
    """
    Reference: 
        https://docs.python.org/3/library/ast.html

    """
    def __init__(self, code_base_path, file_path):
        self.code_base_path = os.path.abspath(code_base_path)
        self.file_name = file_path
        self.module_path = get_module_path(file_path, code_base_path)  # Path of the analyzed file relative to the code base
        self.current_scope = []  # Current scope in the analyzed file (works when function is defined in a class)
        self.function_calls = defaultdict(set)
        self.import_dict = defaultdict(list)
        self.function_defs = set() # To distinguish functions and variables
        self.class_defs = defaultdict(list)


    def analyze_python_file(self):
        """
        Analyze the calling relationship in a single file
        """
        with open(self.file_name, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read())
            self.process_subnodes(tree)
        
        imports = set()
        for _, method in self.import_dict.items():
            imports.add(method)

        class_defs = set()
        for class_def, _ in self.class_defs.items():
            class_defs.add(f"{self.module_path}." + class_def)

        return self.module_path, self.function_calls, imports, self.function_defs, class_defs
    
    def find_import_path(self, import_func, node):
        '''
        Returns the path of the import_func relative to the code base. If the import_func is 
        not imported from the code_base, the function will return module_name.

        E.g.,
        In father_path/file1.py:
        from dir.module_name import import_func

        the function will return dir.import_file
        '''

        current_dir = self.code_base_path

        if isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module
            else:
                module_name = ''

            if node.level > 0:
                current_dir = os.path.dirname(os.path.abspath(self.file_name))
                for _ in range(node.level):
                    current_dir = os.path.dirname(current_dir)

        else:
            module_name = import_func

        
        import_path = module_name
        
        # Convert module name to file path
        module_path = os.path.join(current_dir, *module_name.split('.')) + '.py'
        
        # Check is the module is an existing file
        if os.path.exists(module_path):
            return import_path

        # If not file, check whether it is a package (dir + __init__.py)
        module_path = os.path.join(current_dir, *module_name.split('.'), '__init__.py')
        if not os.path.exists(module_path):
            return import_path

        relative_module_path = os.path.dirname(os.path.relpath(module_path, self.code_base_path))
        relative_module_path = relative_module_path.replace(os.path.sep, ".")

        with open(module_path, 'r') as f:
            file_content = f.read()
            tree = ast.parse(file_content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name == import_func or (alias.asname and alias.asname == import_func):
                            module_name = node.module if node.module else ''

        
        parts = module_name.split(".", 1)
        if relative_module_path:
            if len(parts) > 1:
                import_path = f"{relative_module_path}.{parts[1]}"
            else:
                import_path = f"{relative_module_path}"


        return import_path
    

    def get_target_name(self, node):
        if isinstance(node, ast.Name):
            if node.id == "self":
                return ["unknown"]
            return [node.id]
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                return [node.attr]
            return self.get_target_name(node.value)
        elif isinstance(node, ast.Subscript):
            return self.get_target_name(node.value)
        elif isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            return sum([self.get_target_name(e) for e in node.elts], [])
        else:
            return ["unknown"]


    def get_func_name(self, node):
        if isinstance(node, ast.Name):
            return [node.id]
        elif isinstance(node, ast.Attribute):
            return self.get_func_name(node.value)
        elif isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            return sum([self.get_func_name(e) for e in node.elts], [])
        return ["unknown"]


    def process_classdef(self, node):
        # print(f"Processing classdef")
        self.current_scope.append(node.name)
        # print(f"current_scope: {self.current_scope}")
        self.class_defs[node.name] = []
        
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == 'Function':
                if isinstance(base.value, ast.Attribute) and base.value.attr == 'autograd':
                    if isinstance(base.value.value, ast.Name) and base.value.value.id == 'torch':
                        # Add mapping from class_name.apply to class_name.forward in this special case
                        caller_scope = f""
                        caller_source = f"{self.module_path}." + ".".join(self.current_scope) + ".apply"
                        caller = get_full_function_path(caller_scope, caller_source)

                        callee_scope = caller_source
                        callee_source = f"{self.module_path}." + ".".join(self.current_scope) + ".forward"
                        callee = get_full_function_path(callee_scope, callee_source)
                        self.function_calls[caller].add(callee)
                        # print(f"Add {caller} -> {callee}")

                        self.function_defs.add(f"{node.name}.apply")
                        self.class_defs[node.name].append("apply")

        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.FunctionDef):
                function = self.process_funcdef(sub_node)
                caller_scope = f""
                caller_source = f"{self.module_path}." + ".".join(self.current_scope)
                caller = get_full_function_path(caller_scope, caller_source)
                self.function_calls[caller].add(function)
                # print(f"Add {caller} -> {function}")

        self.current_scope.pop()
        # print(f"current_scope: {self.current_scope}")



    def process_funcdef(self, node):
        # print(f"Processing funcdef")
        caller_scope = f""

        if len(self.current_scope) > 0 and self.current_scope[-1] in self.class_defs:
            self.function_defs.add(".".join([self.current_scope[-1], node.name]))
            self.class_defs[self.current_scope[-1]].append(node.name)
        else:
            self.function_defs.add(node.name)
        
        self.current_scope.append(node.name)
        # print(f"current_scope: {self.current_scope}")
        caller_source = f"{self.module_path}." + ".".join(self.current_scope)

        function = get_full_function_path(caller_scope, caller_source)

        # Get functions called by the node
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call):
                self.process_call(sub_node, function)
            elif isinstance(sub_node, ast.Assign) or isinstance(sub_node, ast.AugAssign):
                self.process_assign(sub_node)
        
    
        # processed_functions.add(node)
        self.current_scope.pop()
        # print(f"current_scope: {self.current_scope}")

        return function

    
    # def process_c_funcdef(self, node, current_scope, function_calls, import_dict, processed_functions):
    #     module_path = self.get_c_module_path(node)
    #     function = get_full_function_path(module_path, current_scope + [node.spelling])

    #     # Get functions called by the node
    #     for sub_node in node.get_children():
    #         if sub_node.kind == clang.cindex.CursorKind.CALL_EXPR:
    #             self.process_c_call(sub_node, function, function_calls, import_dict)
    
    #     processed_functions.add(node)

    def process_func_var_arg(self, arg):
        function = arg.id

        if function == "self":
            return None

        if function in self.import_dict:
            # Imported method
            if len(self.current_scope) > 0:
                scope = f"{self.module_path}." + ".".join(self.current_scope)
            else:
                scope = f"{self.module_path}"
            source = f"{self.import_dict[function]}"
        elif function in self.function_defs:
            # Method defined in current file
            if len(self.current_scope) > 0:
                scope = f"{self.module_path}." + ".".join(self.current_scope)
            else:
                scope = f"{self.module_path}"
            source = f"{self.module_path}.{function}"
        else:
            # Variable
            scope = f"{self.module_path}." + ".".join(self.current_scope[:-1])
            source = f"{self.module_path}." + ".".join(self.current_scope[:-1]) + f".{function}"

        function_full_path = get_full_function_path(scope, source)

        return function_full_path



    def process_args(self, node, caller_function):
        # print(f"Processing args")
        for arg in node.args:
            if isinstance(arg, ast.Name):
                called_function = self.process_func_var_arg(arg)
                if called_function:
                    self.function_calls[caller_function].add(called_function)
                    # print(f"Add {caller_function} -> {called_function}")

            elif isinstance(arg, ast.Attribute):
                if isinstance(arg.value, ast.Name):
                    called_function = self.process_class_call(arg)
                    self.function_calls[caller_function].add(called_function)
                    # print(f"Add {caller_function} -> {called_function}")

            elif isinstance(arg, ast.Call):
                self.process_call(arg, caller_function)

    
    def process_call(self, called_node, caller_function):
        '''
        Process function calls.

        called_node: node of the function call
        caller_function: the function/variable calling the called_node
        '''
        # print(f"Processing call")
        called_function = None
        if isinstance(called_node.func, ast.Name):
            called_function = self.process_func_var_call(called_node.func)

        elif isinstance(called_node.func, ast.Attribute):
            if isinstance(called_node.func.value, ast.Name):
                called_function = self.process_class_call(called_node.func)

        if called_function:
            self.function_calls[caller_function].add(called_function)
            # print(f"Add {caller_function} -> {called_function}")

            if isinstance(called_node.func, ast.Name) and ((called_node.func.id in self.function_defs) or (called_node.func.id in self.import_dict)):
                self.current_scope.append(called_node.func.id)
                # print(f"current_scope: {self.current_scope}")
                self.process_args(called_node, called_function)
                self.current_scope.pop()
                # print(f"current_scope: {self.current_scope}")

            elif isinstance(called_node.func, ast.Attribute):
                obj = called_node.func.value.id
                method = called_node.func.attr
                if obj == "self":
                    # Substitute the self scope with class name
                    obj = self.current_scope[0]
                self.current_scope.append(f"{obj}.{method}")
                # print(f"current_scope: {self.current_scope}")
                self.process_args(called_node, called_function)
                self.current_scope.pop()
                # print(f"current_scope: {self.current_scope}")

            else:
                self.process_args(called_node, called_function)
                
      
    # def process_c_call(self, called_node, caller_function, function_calls, import_dict):
    #     module_path = self.get_c_module_path(called_node)
        
    #     called_function = get_full_function_path(module_path, [called_node.spelling])

    #     if called_function:
    #         function_calls[caller_function].append(called_function)

    #         for arg in called_node.args:
    #             if isinstance(arg, ast.Name) and ((arg.id in function_calls) or (arg.id in import_dict)):
    #                 called_function_2 = self.process_func_var_call(arg, module_path, import_dict)
    #                 function_calls[called_function].append(called_function_2)

    #             elif isinstance(arg, ast.Attribute):
    #                 if isinstance(arg.value, ast.Name):
    #                     called_function_2 = self.process_class_call(arg, module_path, import_dict)
    #                     function_calls[called_function].append(called_function_2)

    #             elif isinstance(arg, ast.Call):
    #                 self.process_call(arg, called_function, module_path, function_calls, import_dict)
                

    
    def process_func_var_call(self, func):

        # print(f"Processing func_call")
        function = func.id
        if len(self.current_scope) > 0:
            scope = f"{self.module_path}." + ".".join(self.current_scope)
        else:
            scope = f"{self.module_path}"

        if function in self.import_dict:
            # Imported method
            source = f"{self.import_dict[function]}"
        elif function in self.function_defs:
            # Method defined in current file
            source = f"{self.module_path}.{function}"
        else:
            # Variable
            source = f"{self.module_path}." + ".".join(self.current_scope) + f".{function}"

        function_full_path = get_full_function_path(scope, source)

        return function_full_path

    
    def process_class_call(self, func):
        # print(f"Processing class_call")
        obj_or_class_name = func.value.id
        method_name = func.attr
        if len(self.current_scope) > 0:
            scope = f"{self.module_path}." + ".".join(self.current_scope)
        else:
            scope = f"{self.module_path}"

        if obj_or_class_name in self.import_dict:
            # Calling imported method
            source = f"{self.import_dict[obj_or_class_name]}." + ".".join([method_name])
        elif obj_or_class_name in self.class_defs:
            # Caling method in class defined in current file
            source = f"{self.module_path}." + ".".join([obj_or_class_name, method_name])
        elif obj_or_class_name == "self":
            # Calling method defined by current class
            source = f"{self.module_path}." + ".".join(self.current_scope[:-1]) + f".{method_name}"
        else:
            # Calling an instance {obj_or_class_name}
            source = f"{self.module_path}." + ".".join(self.current_scope) + f".{obj_or_class_name}"
        
        
        function_full_path = get_full_function_path(scope, source)

        return function_full_path
    
    def get_c_module_path(self, node):
        rel_path = os.path.relpath(node.location.file, start=self.code_base_path)
        file_name, file_ext = os.path.splitext(rel_path)
        file_suffix = file_ext[1:]
        return os.path.join(file_name + '_' + file_suffix)
    
    def process_assign(self, node):
        # print(f"Processing assign")
        if len(self.current_scope) > 0:
            scope = f"{self.module_path}." + ".".join(self.current_scope)
        else:
            scope = f"{self.module_path}"

        try:
            targets = node.targets
        except:
            targets = [node.target]
        
        for target in targets:
            target_name = self.get_target_name(target)
            for i in range(len(target_name)):
                caller = get_full_function_path(scope, f"{scope}.{target_name[i]}")
                if len(self.current_scope) > 0:
                    outer_scope = "" if len(self.current_scope) == 1 else f"{self.module_path}." + ".".join(self.current_scope[:-1])
                    caller_scope = get_full_function_path(outer_scope, scope)
                    self.function_calls[caller_scope].add(caller)

                if isinstance(node.value, ast.Name):  # right value is a variable
                    callee = get_full_function_path(scope, f"{scope}." + node.value.id)
                    self.function_calls[caller].add(callee)
                    # print(f"Add {caller} -> {callee}")


                elif isinstance(node.value, ast.Call):  # right value is a function call or class call
                    self.process_call(node.value, caller)
                    

                elif isinstance(node.value, ast.BinOp):  # right value is an expression
                    left_name = self.get_target_name(node.value.left)
                    right_name = self.get_target_name(node.value.right)
                    called_vars = left_name + right_name
                    callee = [get_full_function_path(scope, f"{scope}.{e}") for e in called_vars]
                    self.function_calls[caller].update(callee)
                    # print(f"Add {caller} -> {callee}")


                elif isinstance(node.value, ast.Attribute) or isinstance(node.value, ast.Subscript):  # right value is an attr or index
                    obj_names = self.get_target_name(node.value.value)
                    obj_name = obj_names[0] if len(obj_names) == 1 else obj_names[i]

                    callee = get_full_function_path(scope, f"{scope}.{obj_name}")
                    self.function_calls[caller].add(callee)
                    # print(f"Add {caller} -> {callee}")

    
    
    def process_subnodes(self, tree):
        if not tree.body:
            return
        
        for node in tree.body:
            bool_call_in_val = (isinstance(node, ast.Expr) or isinstance(node, ast.Return)) and node.value and isinstance(node.value, ast.Call)
            bool_call_in_test = (isinstance(node, ast.For) and node.iter and isinstance(node.iter, ast.Call)) or ((isinstance(node, ast.While) or isinstance(node, ast.If)) and node.test and isinstance(node.test, ast.Call))
            bool_have_call = isinstance(node, ast.Call) or bool_call_in_val or bool_call_in_test

            # Get imported modules
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # alias.name is the original module, alias.asname is the alias (if present)
                    module_name = self.find_import_path(alias.name, node)
                    full_name = f"{module_name}".strip(".")
                    if alias.asname:
                        self.import_dict[alias.asname] = full_name
                    else:
                        self.import_dict[alias.name] = full_name
            
            # Handle "from c import d as e"
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    module_name = self.find_import_path(alias.name, node)
                    full_name = f"{module_name}.{alias.name}".strip(".")
                    if alias.asname:
                        self.import_dict[alias.asname] = full_name
                    else:
                        self.import_dict[alias.name] = full_name
            
            elif isinstance(node, ast.ClassDef):
                # Can detect the functions defined in Classes here
                self.process_classdef(node)
                
            elif isinstance(node, ast.FunctionDef):
                self.process_funcdef(node)

            elif bool_have_call:
                if isinstance(node, ast.If) or isinstance(node, ast.While):
                    node = node.test
                elif isinstance(node, ast.For):
                    node = node.iter
                elif isinstance(node, ast.Call):
                    node = node
                else:
                    node = node.value

                function = None
                if isinstance(node.func, ast.Name):
                    function = self.process_func_var_call(node.func)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        function = self.process_class_call(node.func)

                if function:
                    for arg in node.args:
                        if isinstance(arg, ast.Name) or isinstance(arg, ast.Attribute):
                            if isinstance(arg, ast.Name) and ((arg.id in self.function_calls) or (arg.id in self.import_dict)):
                                called_function = self.process_func_var_call(arg)
                                self.function_calls[function].add(called_function)
                                # print(f"Add {function} -> {called_function}")
                            elif isinstance(arg, ast.Attribute):
                                if isinstance(arg.value, ast.Name) and ((arg.value.id in self.function_calls) or (arg.value.id in self.import_dict)):
                                    called_function = self.process_class_call(arg)
                                    self.function_calls[function].add(called_function)
                                    # print(f"Add {function} -> {called_function}")

                        elif isinstance(arg, ast.Call):
                            self.process_call(arg, function)

            elif isinstance(node, ast.Assign) or isinstance(node, ast.AugAssign):
                self.process_assign(node)

            else:
                try:
                    self.process_subnodes(node)
                except:
                    return 
        



    
if __name__ == "__main__":
    input_path = input("Path to the code base: ")
    
    output_json = False
    output_chain = False
    filter_cuda = False
    
    mode = input("Output mode: [json/chain]\n")
    if mode == "json":
        output_json = True
    elif mode == "chain":
        output_chain = True
    else:
        print("Please type in \"json\" or \"chain\"\n")
        exit()
        
    cuda = input("Only output cuda-related chains? [Y/N]\n")
    if cuda == "Y" or cuda == "y":
        filter_cuda = True
    elif cuda == "N" or cuda == "n":
        filter_cuda = False
    else:
        print("Please type in \"Y\" or \"N\"\n")
        exit()
    
    get_relationship_map_in_whole(input_path, output_json=output_json, output_chain=output_chain, filter_cuda=filter_cuda)
    # get_relationship_map(input_path, output_json, output_chain)

