import ast
import textwrap
from fem_bench.task_base import Task


def extract_signature_and_docstring(code: str) -> tuple[str, str]:
    """Extract function signature and formatted docstring from a code string."""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_node = node
            break
    else:
        raise ValueError("No function definition found.")

    args = []
    for arg in func_node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        args.append(arg_str)

    args_str = ", ".join(args)
    return_str = f" -> {ast.unparse(func_node.returns)}" if func_node.returns else ""
    signature = f"def {func_node.name}({args_str}){return_str}:"

    docstring = ast.get_docstring(func_node) or ""
    docstring_indented = textwrap.indent(f'"""\n{docstring}\n"""', "    ")

    return signature, docstring_indented


def format_dependency_functions(dep_code_list: list[str]) -> str:
    """Format dependency function signatures and docstrings."""
    if not dep_code_list:
        return ""

    blocks = []
    for code in dep_code_list:
        try:
            sig, doc = extract_signature_and_docstring(code)
            blocks.append(f"{sig}\n{doc}")
        except Exception:
            continue  # skip if unparsable

    if not blocks:
        return ""

    return "## Available Helper Functions:\n" + "\n\n".join(blocks) + "\n"


def task_to_code_prompt(task: Task) -> str:
    """Generate a structured LLM prompt with signature, docstring, imports, and helpers."""
    signature, docstring = extract_signature_and_docstring(task.main_fcn_code)

    # Import constraints
    if task.required_imports:
        import_lines = "\n".join(task.required_imports)
        import_instruction = f"- Use only the following imports: {import_lines}"
    else:
        import_instruction = "- No imports are available"

    # Dependencies (helper functions)
    helper_section = format_dependency_functions(task.fcn_dependency_code)

    prompt = f"""\
# Python Function Implementation Task

Write a Python function that matches the exact signature and docstring provided below.

## Requirements:
- Keep the function name, parameter names, and docstring exactly as shown
- Do not add any code outside the function definition
{import_instruction}
- You may call only the helper functions listed below (if any)
- Output only valid Python code (no explanations, comments, or markdown)
- Implement the functionality as described in the docstring

{helper_section}## Function Signature:
{signature}
{docstring}
"""
    return prompt


def extract_test_name_and_docstring(code: str) -> tuple[str, str]:
    """Extract the function name and docstring from a test function code block."""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            doc = ast.get_docstring(node) or ""
            return name, doc
    raise ValueError("No test function found.")


def task_to_test_prompt(task: Task) -> str:
    """Generate a prompt that instructs the LLM to write pytest tests for the given task."""
    # Extract main function signature and docstring
    try:
        signature, docstring = extract_signature_and_docstring(task.main_fcn_code)
    except Exception:
        signature, docstring = "def <unknown>():", '    """Missing docstring."""'

    # Extract test names and their docstrings
    test_lines = []
    for case in task.test_cases:
        try:
            tree = ast.parse(case["test_code"])
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    name = node.name
                    doc = ast.get_docstring(node) or "(no description)"
                    test_lines.append(f"- {name}: \"{doc.strip()}\"")
        except Exception:
            continue

    if not test_lines:
        test_block = "- (no test cases found)"
    else:
        test_block = "\n".join(test_lines)

    prompt = f"""\
# Python Task: Write Pytest Tests for a Function

Below is the function you are testing. Use its signature and docstring to understand its behavior.

## Function Signature:
{signature}
{docstring}

## Your Goal:
Write pytest-style test functions that verify the correctness of the function above.

## Requirements:
- Use the exact test function names listed below
- Each test must accept a single argument: `fcn` — the function to test
- Use `assert` statements to check correctness
- Each test must include a descriptive docstring
- Do not include print statements, logging, or example usage
- Output only valid Python code — no explanations, markdown, or comments

## Test Functions to Implement:
{test_block}

# Output:
# One or more valid pytest functions matching the names and purposes above
"""
    return prompt
