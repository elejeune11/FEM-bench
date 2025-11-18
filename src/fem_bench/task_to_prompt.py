import ast
import textwrap
from fem_bench.task_base import Task
from jinja2 import Environment, FileSystemLoader


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
    docstring_clean = textwrap.dedent(docstring).strip()
    docstring_indented = textwrap.indent(f'"""\n{docstring_clean}\n"""', "    ")

    return signature, docstring_indented


def task_to_code_prompt(task: Task, template_dir: str, template_name: str) -> str:
    """Generate a structured LLM prompt with signature, docstring, imports, and helpers."""
    env = Environment(
        loader=FileSystemLoader(searchpath=template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    code_template = env.get_template(template_name)
    signature, docstring = extract_signature_and_docstring(task.main_fcn_code)

    context = {
        "task": task,
        "signature": signature,
        "docstring": docstring,
    }

    return code_template.render(context)


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

## Only complete the test functions below:
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

## Function Signature:
## Test Functions to Implement:
{test_block}

# Output:
# Only return valid pytest test functions — no prose, markdown, or commentary.
"""
    return prompt
