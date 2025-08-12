# --- clean_output.py (updated) ---
import ast
import re
from typing import Dict, List

DEFAULT_JUNK_PREFIXES = (
    "#",
    "- ",
    "* ",
    "```",
    "Output only",
    "Here's",
    "Below is",
    "The following",
    "This function",
)

def strip_code_fences(text: str) -> str:
    """
    Removes common markdown-style code fences from the text.
    """
    lines = text.strip().splitlines()
    while lines and re.match(r"^\s*```(?:python)?", lines[0], re.IGNORECASE):
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def remove_junk_lines(text: str, junk_prefixes: tuple = DEFAULT_JUNK_PREFIXES) -> str:
    """
    Remove comment lines, markdown bullets, and known LLM chatter.
    """
    lines = text.strip().splitlines()
    return "\n".join(
        line for line in lines
        if line.strip() and not any(line.strip().lower().startswith(prefix.lower()) for prefix in junk_prefixes)
    ).strip()

def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def extract_first_function(code: str) -> str:
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return ast.unparse(node)
    except Exception as e:
        print(f"Error during AST extraction: {e}")
    raise ValueError("No function definition found.")

def extract_all_functions(code: str) -> Dict[str, str]:
    result = {}
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                result[node.name] = ast.unparse(node)
    except Exception as e:
        print(f"AST error: {e}")
    return result

def fallback_extract_by_regex(text: str) -> str:
    match = re.search(r"(?s)^def\s+\w+\(.*?\):\s*(\"\"\".*?\"\"\"|'''.*?''')?[\s\S]*?(?=^def\s|\Z)", text)
    if match:
        return match.group(0).strip()
    return text.strip()

def clean_and_extract_function(llm_output: str) -> str:
    # print("=== RAW LLM OUTPUT ===")
    # print(llm_output)  # Log before anything else
    text = strip_code_fences(llm_output)
    text = remove_junk_lines(text)
    if not is_valid_python(text):
        return fallback_extract_by_regex(text)
    return extract_first_function(text)

def extract_test_functions(source: str) -> Dict[str, str]:
    cleaned = strip_code_fences(source)
    cleaned = remove_junk_lines(cleaned)
    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        return extract_test_functions_fallback(cleaned)
    return {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }

def extract_test_functions_fallback(source: str) -> Dict[str, str]:
    blocks = source.split("\n\n")
    result = {}
    for block in blocks:
        block = block.strip()
        if block.startswith("def test_"):
            try:
                tree = ast.parse(block)
                func = tree.body[0]
                if isinstance(func, ast.FunctionDef):
                    result[func.name] = ast.unparse(func)
            except Exception:
                continue
    return result
