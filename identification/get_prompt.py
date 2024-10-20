import os
import re

from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser


def get_prompt(chain, start_file, other_func):
    prompt = (
        "The following is a function call chain starting from the code file listed below. The code base is an implementation of the deployment of Transformer.\n\n"
        "Identify all **hyperparameters** along the call chain and extract those are **related to the arguments of `flash_attn_2_cuda`** (the end of the call chain).\n"
        "- Focus on **hyperparameters** that are related to **attention score computation** during **attention mechanism**, especially for **Transformers**.\n"
        "- Identify hyperparameters that **affect the model's architecture, behavior, and prediction quality**, especially those related to **attention score**.\n"
        "- Identify **hyperparameters** determining the **sizes, dimensions, and shapes** of tensors input to `flash_attn_2_cuda`\n"
        "- **Ignore** hyperparameters that are **solely used to optimize computational efficiency** and **do not impact the model's output**, such as parameters related to **memory management, caching mechanisms, or computational acceleration**.\n"
        "- **Keep** all hyperparameters that will influence **attention model's output performance**.\n"
        "- Ignore the boolean hyperparameters that only decide return format of a function but do nothing with the attention mechanism.\n\n\n"
        "**Function call chain**:\n"
        f"`{chain}`\n\n"
        "Each function/variable is formatted as: the part before \"/\" indicates the scope of the function/variable being called, and the part after \"/\" indicates where the function originates from.\n\n\n"
        "**Code in the scope of starting point**:\n\n"
        f"{start_file}\n"
        "**Other functions in the call chain**:\n\n"
        )
    
    for function in other_func:
        prompt += function
        
    if len(other_func) == 0:
        prompt += "None\n\n\n\n"
        
    prompt += (
        "Based on your knowledge of Transformer, identify all **hyperparameters** in the code snippets that finally related to `flash_attn_2_cuda`, especially for those hyperparameters determining the input tensors' **sizes, dimensions, and shapes**.\n"
        "Focus on **hyperparameters** that are related to **attention score computation** during **training or inference** with **attention mechanism**, especially for **Transformers**.\n"
        "You should:\n"
        "- Identify hyperparameters that **affect the model's architecture, behavior, and prediction quality**, especially those related to **attention score**.\n"
        "- Identify **hyperparameters** determining the **sizes, dimensions, and shapes** of tensors input to `flash_attn_2_cuda`, or those determine the attention model structure in `flash_attn_2_cuda`.\n"
        "- **Ignore** hyperparameters that are **solely used to optimize computational efficiency** and do not impact the model's output, such as parameters related to **memory management, caching mechanisms, or computational acceleration**.\n"
        "- **Keep** all hyperparameters that will influence **attention model's output performance**.\n"
        "- Ignore the boolean hyperparameters that only decide return format of a function but do nothing with the attention mechanism.\n\n"
        
        "Provide a brief description of each parameter you identified, including\n"
        "- **Scope** in which the parameter is used. Show the scope in the following format: Join the path to where it is from directory to function using \".\".\n"
        "  If a parameter appears in multiple functions, use \", \" to separate different scope paths. E.g., if parameter `a` appears in `func1` in `flash_attn/file1.py` and `func2` in `flash_attn/file2.py`, its scope should be written as: `flash_attn.file1.func1, flash_attn.file2.func2`\n"
        "- The role or purpose of the parameter\n\n"

        "Only output a **JSON**. Set the keys to hyperparameters you identified and values to corresponding scopes and descriptions. Put all your output in the JSON structure with the following format:\n"
        "```json\n"
        "{\n"
        "    \"hyperparameter1\":\"[scope1] description1\",\n"
        "    \"hyperparameter2\":\"[scope2] description2\"\n"
        "}\n"
        "```\n"
        )

    return prompt


def extract_file_path(code_base, call, source):
    scope_file, source_file = call.split("/", 1)
    if source:
        function_file = source_file
    else:
        function_file = scope_file
    
    parts = function_file.split(".")
    
    file_path = code_base
    for part in parts:
        temp_path = os.path.join(file_path, part) + '.py'
        if os.path.exists(temp_path):
            file_path = temp_path
            break
        
        file_path = os.path.join(file_path, part)
        
    if os.path.exists(file_path):
        return file_path
    else:
        return None


def extract_file_content(file_path):
    file_content = "\n"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            file_content = f.read()
        file_content = "```python\n" + file_content + "\n```\n\n\n"
            
    return file_content
    

def extract_func_name(file_path, call):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    _, function_path = call.split("/", 1)
    names = function_path.split(".")
    for i, name in enumerate(names):
        if name == file_name and i + 1 < len(names):
            return file_name, names[i + 1]
    
    return file_name, None

def extract_func_fullname(file_path, call, source):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    scope_path, source_path = call.split("/", 1)
    if source:
        function_path = source_path
    else:
        function_path = scope_path

    names = function_path.split(".")
    for i, name in enumerate(names):
        if name == file_name and i + 1 < len(names):
            return ".".join(names[:i + 2])
        
    return None
            

def extract_func_or_class(file_path, call):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    _, name = extract_func_name(file_path, call)
    if not name:
        return None
    

    loader = GenericLoader.from_filesystem(
        file_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON)
    )

    docs = loader.load()

    definition = None
    for doc in docs:
        if doc.page_content and doc.metadata['content_type'] == 'functions_classes':
            if doc.page_content.strip().startswith(f"def {name}") or doc.page_content.strip().startswith(f"class {name}"):
                definition = doc.page_content
        
    if definition:
        return "```python\n" + definition + "\n```\n\n"
    else:
        return None


def get_input(callgraphchain, code_base):
    os.makedirs("prompt", exist_ok=True)
    
    with open(callgraphchain, 'r', encoding='utf-8') as f:
        file_content = f.read()

    call_chains = file_content.strip().split("\n\n")

    i = 0
    for chain in call_chains:
        i += 1
        calls = chain.split(" -> ")
        
        other_function = []
        added_function = set()

        start_call = calls[0]
        start_scope = start_call.split("/")[0]

        if start_scope == "":
            start_file_path = extract_file_path(code_base, start_call, True)
            start_func = extract_func_fullname(start_file_path, start_call, True)
            start_file_content = extract_func_or_class(start_file_path, start_call)
            if start_func:
                added_function.add(start_func)
        else:
            start_file_path = extract_file_path(code_base, start_call, False)
            start_func = start_scope
            start_file_content = extract_file_content(start_file_path)

        start_file_content = f"`{start_func}`:\n" + start_file_content
        
        for call in calls[1:]:
            file_path = extract_file_path(code_base, call, True)
            if not file_path:
                continue
            if file_path == start_file_path and start_scope != "":
                continue
            
            func = extract_func_fullname(file_path, call, True)
            if func:
                if func in added_function:
                    continue
                added_function.add(func)
                
                function = f"`{func}`:\n" + extract_func_or_class(file_path, call)
                if function:
                    other_function.append(function)

        prompt = get_prompt(chain, start_file_content, other_function)
        output_file = os.path.join("prompt", f"prompt_api_{i}")
        with open(output_file, 'w') as f:
            f.write(prompt)



if __name__ == '__main__':
    code_base = input("Path to code base:")
    callgraphchain = input("Path to call graph chain file:")
    
    code_base = os.path.abspath(code_base)
    callgraphchain = os.path.abspath(callgraphchain)
    
    get_input(callgraphchain, code_base)
