"""
Prompt builder for converting FEM-Bench tasks into LLM prompts.
"""
from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path
import re
from typing import Dict, List, NamedTuple, Union, Any

from fem_bench.yaml_load import load_task


# --------------------------------------------------------------------------- #
#                           Helper: type normalisation                        #
# --------------------------------------------------------------------------- #
def _norm_type(t: str | None) -> str:
    """Map aliases & strip whitespace so 'numpy.ndarray' == 'np.ndarray' etc."""
    if not t:
        return "any"
    return (
        t.replace("numpy.", "np.")
         .replace(" ", "")
         .lower()
         .removeprefix("typing.")          # tolerate typing aliases
    )


def estimate_token_count(prompt: str) -> int:
    """
    Rough estimation of token count for the prompt.
    Uses approximation of ~4 characters per token.
    
    Args:
        prompt: The prompt string
        
    Returns:
        Estimated token count
    """
    return len(prompt) // 4


def build_dependency_code(task, task_dir=None):
    """
    Build dependency information string by loading YAML files for each dependency.
    
    Args:
        task: Task object with task_dependencies dict
        task_dir: Directory to look for task files (optional, defaults to current dir)
        
    Returns:
        Nicely formatted dependency info for the prompt
    """
    if not task.task_dependencies or "required_functions" not in task.task_dependencies:
        return ""

    dependency_sections = []

    for dep in task.task_dependencies["required_functions"]:
        # Use attribute access (not dict) since dep is a TaskDependency model
        task_file = Path(task_dir or ".") / f"{dep.source_task}.yaml"
        source_task = load_task(task_file)

        function_name = source_task.expected_function_name
        description = source_task.short_description
        signature = source_task.function_signature

        # Build parameter list: name: type
        param_strs = [
            f"{param.name}: {param.type}"
            for param in signature.input_parameters
        ]

        # Return type: handle single or multiple return values
        if len(signature.return_parameters) == 1:
            return_type = signature.return_parameters[0].type
        else:
            return_type = f"Tuple[{', '.join(p.type for p in signature.return_parameters)}]"

        # Full function signature
        full_signature = f"def {function_name}({', '.join(param_strs)}) -> {return_type}"

        # Format for prompt
        dependency_sections.append(f"**{function_name}** (from {dep.source_task}):")
        dependency_sections.append(f"- Description: {description}")
        dependency_sections.append(f"- Signature: `{full_signature}`")
        dependency_sections.append("")  # blank line

    return "\n".join(dependency_sections).strip()

# Original version -- now depreciated
# def build_prompt(task, environment, task_dir=None):
#     """
#     Build a complete LLM prompt from task and environment specifications.
    
#     Args:
#         task: Task object loaded from YAML
#         environment: Environment object loaded from YAML
#         task_dir: Directory containing task dependency files (optional)
        
#     Returns:
#         Complete formatted prompt string for the LLM
#     """
#     prompt_sections = []
    
#     # ============================================================================
#     # ENVIRONMENT INFORMATION
#     # ============================================================================
    
#     prompt_sections.append("# Environment Configuration")
#     prompt_sections.append(f"**Environment**: {environment.environment_name} (Tier {environment.tier})")
#     prompt_sections.append(f"**Description**: {environment.description}")
#     prompt_sections.append(f"**Language**: {environment.language} {environment.python_version}")
#     prompt_sections.append("")
    
#     # Required Libraries
#     prompt_sections.append("## Required Libraries")
#     prompt_sections.append("The following libraries are always available and should be used:")
#     for lib in environment.required_libraries:
#         import_str = f" (import as `{lib.import_as}`)" if lib.import_as else ""
#         prompt_sections.append(f"- **{lib.name}** {lib.version}{import_str}: {lib.purpose}")
#     prompt_sections.append("")
    
#     # Allowed Libraries
#     if environment.allowed_libraries:
#         prompt_sections.append("## Allowed Libraries")
#         prompt_sections.append("Additional libraries that can be used when needed:")
#         for lib in environment.allowed_libraries:
#             import_str = f" (import as `{lib.import_as}`)" if lib.import_as else ""
#             usage_str = f" - {lib.usage}" if hasattr(lib, 'usage') and lib.usage else ""
#             prompt_sections.append(f"- **{lib.name}** {lib.version}{import_str}: {lib.purpose}{usage_str}")
#         prompt_sections.append("")
    
#     # Import Guidelines
#     if environment.import_guidelines:
#         prompt_sections.append("## Import Guidelines")
#         prompt_sections.append(environment.import_guidelines)
#         prompt_sections.append("")
    
#     # Code Requirements
#     if environment.code_requirements:
#         prompt_sections.append("## Code Requirements")
#         for req_name, req_value in environment.code_requirements.items():
#             req_display = req_name.replace('_', ' ').title()
#             prompt_sections.append(f"- **{req_display}**: {req_value}")
#         prompt_sections.append("")
    
#     # Testing Requirements
#     if environment.testing:
#         prompt_sections.append("## Testing Requirements")
#         testing = environment.testing
#         if 'framework' in testing:
#             prompt_sections.append(f"- **Framework**: {testing['framework']}")
#         if 'naming_convention' in testing:
#             prompt_sections.append(f"- **Test Function Naming**: {testing['naming_convention']}")
#         if 'required_imports' in testing:
#             imports = ", ".join(testing['required_imports'])
#             prompt_sections.append(f"- **Required Test Imports**: {imports}")
#         prompt_sections.append("")
    
#     # ============================================================================
#     # TASK DEPENDENCIES
#     # ============================================================================
    
#     dependency_code = build_dependency_code(task, task_dir=task_dir)
#     if dependency_code:
#         prompt_sections.append("# Task Dependencies")
#         prompt_sections.append("Your implementation can use the following functions from previous tasks:")
#         prompt_sections.append("")
#         prompt_sections.append(dependency_code)
#         prompt_sections.append("")
    
#     # ============================================================================
#     # TASK SPECIFICATION
#     # ============================================================================
    
#     prompt_sections.append("# Task Specification")
#     prompt_sections.append(f"**Task**: {task.title}")
#     prompt_sections.append(f"**Description**: {task.short_description}")
#     prompt_sections.append("")
    
#     # Function Signature
#     prompt_sections.append("## Required Function")
#     sig = task.function_signature
#     params_with_types = []
#     for param, param_type in zip(sig.parameters, sig.parameter_types):
#         params_with_types.append(f"{param}: {param_type}")
    
#     function_signature = f"def {task.expected_function_name}({', '.join(params_with_types)}) -> {sig.return_shape}"
#     prompt_sections.append(f"**Function Name**: `{task.expected_function_name}`")
#     prompt_sections.append(f"**Signature**: `{function_signature}`")
#     prompt_sections.append("")
    
#     # Expected Test Functions
#     if task.include_tests and task.expected_test_functions:
#         prompt_sections.append("## Required Test Functions")
#         prompt_sections.append("Implement the following test functions:")
#         for test_func in task.expected_test_functions:
#             prompt_sections.append(f"- `{test_func}`")
#         prompt_sections.append("")
    
#     # Main Task Prompt
#     prompt_sections.append("## Task Details")
#     prompt_sections.append(task.prompt)
#     prompt_sections.append("")
    
#     # ============================================================================
#     # OUTPUT FORMAT REQUIREMENTS
#     # ============================================================================
    
#     prompt_sections.append("# Output Format Requirements")
#     prompt_sections.append("")
#     prompt_sections.append("**IMPORTANT**: Your response must be valid JSON with named function fields.")
#     prompt_sections.append("")
#     prompt_sections.append("Return your response in this exact JSON format:")
#     prompt_sections.append("")
#     prompt_sections.append("```json")
#     prompt_sections.append("{")
#     prompt_sections.append('  "function_imports": ["numpy"],')
#     if task.include_tests and task.expected_test_functions:
#         prompt_sections.append('  "test_imports": ["pytest"],')
#     prompt_sections.append(f'  "{task.expected_function_name}": "def {task.expected_function_name}(...):\\n    \\"\\"\\"Docstring\\"\\"\\"\\n    # Your implementation\\n    pass",')
    
#     if task.include_tests and task.expected_test_functions:
#         for test_func in task.expected_test_functions:
#             prompt_sections.append(f'  "{test_func}": "def {test_func}():\\n    \\"\\"\\"Test docstring\\"\\"\\"\\n    # Test implementation\\n    pass",')
    
#     prompt_sections.append("}")
#     prompt_sections.append("```")
#     prompt_sections.append("")
    
#     # JSON Schema
#     prompt_sections.append("## JSON Schema")
#     prompt_sections.append("Your response must include these exact fields:")
#     prompt_sections.append("- `function_imports` (array): List of library names used by the main function (e.g., ['numpy', 'math'])")
#     if task.include_tests and task.expected_test_functions:
#         prompt_sections.append("- `test_imports` (array): List of library names used by test functions (e.g., ['pytest', 'unittest'])")
#     prompt_sections.append(f"- `{task.expected_function_name}` (string): Complete function definition code")
#     if task.include_tests and task.expected_test_functions:
#         for test_func in task.expected_test_functions:
#             prompt_sections.append(f"- `{test_func}` (string): Complete test function definition code")
#     prompt_sections.append("")
    
#     # Function Template
#     prompt_sections.append("## Implementation Template")
#     prompt_sections.append(f"Your `{task.expected_function_name}` field should contain:")
#     prompt_sections.append("")
#     prompt_sections.append(f"```python")
#     prompt_sections.append(f"def {task.expected_function_name}({', '.join(params_with_types)}) -> {sig.return_shape}:")
#     prompt_sections.append('    """')
#     prompt_sections.append(f"    {task.short_description}")
#     prompt_sections.append('    """')
#     prompt_sections.append("    # Your implementation here")
#     prompt_sections.append("    pass")
#     prompt_sections.append("```")
#     prompt_sections.append("")
    
#     # Test Templates if needed
#     if task.include_tests and task.expected_test_functions:
#         prompt_sections.append("## Test Function Templates")
#         for test_func in task.expected_test_functions:
#             prompt_sections.append(f"Your `{test_func}` field should contain:")
#             prompt_sections.append("")
#             prompt_sections.append("```python")
#             prompt_sections.append(f"def {test_func}():")
#             prompt_sections.append('    """Test function implementation."""')
#             prompt_sections.append("    # Your test implementation here")
#             prompt_sections.append("    assert True  # Replace with actual tests")
#             prompt_sections.append("```")
#             prompt_sections.append("")
    
#     # ============================================================================
#     # JSON FORMATTING INSTRUCTIONS
#     # ============================================================================
    
#     prompt_sections.append("# JSON Formatting Rules")
#     prompt_sections.append("1. **Valid JSON only**: Response must be parseable with `json.loads()`")
#     prompt_sections.append("2. **Escape characters**: Use `\\n` for newlines, `\\\"` for quotes in strings")
#     prompt_sections.append("3. **No comments**: JSON does not support comments - put all code in string fields")
#     prompt_sections.append("4. **Exact field names**: Use the function names shown above exactly as field names")
#     prompt_sections.append("5. **Complete function definitions**: Each field contains a complete `def` statement")
#     prompt_sections.append("6. **Separate imports**: Put function imports in `function_imports` and test imports in `test_imports`")
#     prompt_sections.append("7. **No imports in function strings**: Function definitions should not contain import statements")
#     prompt_sections.append("")
    
#     # ============================================================================
#     # FINAL INSTRUCTIONS
#     # ============================================================================
    
#     prompt_sections.append("# Implementation Instructions")
#     prompt_sections.append("1. Return ONLY valid JSON - no additional text before or after")
#     prompt_sections.append("2. Use function names as JSON field names exactly as specified")
#     prompt_sections.append("3. Include main function imports in the `function_imports` array")
#     if task.include_tests:
#         prompt_sections.append("4. Include test function imports in the `test_imports` array")
#         prompt_sections.append("5. Ensure all tests pass with your implementation")
#         prompt_sections.append("6. Follow the environment requirements and coding standards")
#         prompt_sections.append("7. Use only the allowed libraries listed above")
#     else:
#         prompt_sections.append("4. Follow the environment requirements and coding standards")
#         prompt_sections.append("5. Use only the allowed libraries listed above")
    
#     # Join all sections
#     return "\n".join(prompt_sections)


# Existing imports (ast, re, etc.) are still present above in prompt.py
# ---------------------------------------------------------------------------
# --------------------------------------------------------------------------- #
#                          Constant text blocks                                #
# --------------------------------------------------------------------------- #
_FORMATTING_RULES = textwrap.dedent(
    """
    # JSON Formatting Rules
    1. **Valid JSON only** – the assistant’s reply must parse with `json.loads`.
    2. **Escape characters** – use `\\n` for new-lines, `\\"` for quotes *inside* strings.
    3. **No comments** – JSON does not support comments; keep code inside string fields.
    4. **Exact field names** – match the schema exactly (case-sensitive).
    5. **Complete function definitions** – every code field holds a full `def …`.
    6. **Separate imports** – list imports in `function_imports` / `test_imports`,\
       **never** inside code strings.
    """
).strip()

_FORMATTING_WARNING = (
    "⚠️ Put **all imports** in the `function_imports` / `test_imports` arrays – "
    "**do not** write `import …` statements inside any code string."
)

# --------------------------------------------------------------------------- #
#                                Helpers                                      #
# --------------------------------------------------------------------------- #
def _section(title: str, body_lines: List[str]) -> str:
    """Render a Markdown section if body_lines is non-empty."""
    if not body_lines:
        return ""
    return f"# {title}\n" + "\n".join(body_lines) + "\n"


def _libs_to_markdown(libs, heading: str, *, verbosity: str) -> str:
    if not libs:
        return ""
    bullet = (
        lambda lib: f"- **{lib.name}** {lib.version}"
        + (f" (import as `{lib.import_as}`)" if lib.import_as else "")
        + (f": {lib.purpose}" if verbosity == "full" and lib.purpose else "")
    )
    return _section(heading, [bullet(lib) for lib in libs])


def _dict_to_bullets(d: Any, sort_keys: bool = True) -> List[str]:
    """
    Convert a shallow dict or Pydantic model into bullet-point lines.
    Each line is formatted as: - **key**: value

    Parameters:
        d (dict or Pydantic model): The source data.
        sort_keys (bool): Whether to sort keys alphabetically.

    Returns:
        List[str]: A list of bullet lines.
    """
    if hasattr(d, "model_dump"):
        d = d.model_dump()

    if not isinstance(d, dict):
        raise TypeError(f"_dict_to_bullets expects a dict or Pydantic model, got {type(d)}")

    items = d.items()
    if sort_keys:
        items = sorted(items)

    return [f"- **{k}**: {v}" for k, v in items]


def _make_task_specific_skeleton(task) -> dict:
    """Generate a minimal JSON skeleton *for this task*."""
    skel = {
        "function_imports": ["numpy"],
        task.expected_function_name: f"def {task.expected_function_name}(...):\\n    ..."
    }
    if task.include_tests and task.expected_test_functions:
        skel["test_imports"] = ["pytest"]
        for t in task.expected_test_functions:
            skel[t] = f"def {t}(...):\\n    ..."
    return skel


# --------------------------------------------------------------------------- #
#                             Main builder                                    #
# --------------------------------------------------------------------------- #
def build_prompt(
    task,
    environment,
    task_dir: str | Path | None = None,
    *,
    verbosity: str = "full",
) -> str:
    """
    Build the LLM prompt for a FEM-bench task, including environment specs,
    user-supplied description (`task.prompt_description`), and JSON-format instructions.
    """
    from fem_bench.prompt import build_dependency_code  # local import to avoid cycles

    lines: List[str] = []

    # ── 1. Environment configuration ─────────────────────────────────────
    env_info = [
        f"**Environment**: {environment.environment_name} (Tier {environment.tier})",
        f"**Language**: {environment.language} {environment.python_version}",
        f"**Description**: {environment.description}",
    ]
    lines.append(_section("Environment Configuration", env_info))

    # Libraries
    lines.append(_libs_to_markdown(environment.required_libraries, "Required Libraries", verbosity=verbosity))
    if environment.allowed_libraries:
        lines.append(_libs_to_markdown(environment.allowed_libraries, "Allowed Libraries", verbosity=verbosity))

    # Optional environment extras
    if environment.import_guidelines:
        lines.append(_section("Import Guidelines", environment.import_guidelines.strip().splitlines()))

    if environment.code_requirements:
        lines.append(_section("Code Requirements", _dict_to_bullets(environment.code_requirements.model_dump())))

    if environment.testing:
        lines.append(_section("Testing Requirements", _dict_to_bullets(environment.testing.model_dump())))

    # ── 2. Task dependencies (if any) ─────────────────────────────────────
    dep_block = build_dependency_code(task, task_dir=task_dir)
    if dep_block:
        lines.append(_section("Task Dependencies", ["Your implementation can use the following functions:", "", dep_block]))

    # ── 3. Task specification ────────────────────────────────────────────
    sig = task.function_signature
    params = ", ".join(f"{param.name}: {param.type}" for param in sig.input_parameters)
    if len(sig.return_parameters) == 1:
        return_type = sig.return_parameters[0].type
    else:
        return_type = f"Tuple[{', '.join(p.type for p in sig.return_parameters)}]"
    task_spec = [
        f"**Task**: {task.title}",
        f"**Function**: `{task.expected_function_name}`",
        f"**Signature**: `def {task.expected_function_name}({params}) -> {return_type}`",
        task.short_description,
    ]
    lines.append(_section("Task Specification", task_spec))

    # ── 4. Main task description (verbatim from YAML) ────────────────────
    if task.prompt_description:
        lines.append(_section("Main Task Description", task.prompt_description.strip().splitlines()))

    # ── 5. Output-format rules + JSON skeleton ───────────────────────────
    skeleton_dict = _make_task_specific_skeleton(task)
    json_example = json.dumps(skeleton_dict, indent=2)

    lines.append(
        _section(
            "Output Format Requirements",
            [
                "**Respond with valid JSON – nothing else.**",
                "Example:",
                f"```json\n{json_example}\n```",
                _FORMATTING_WARNING,
                _FORMAT_WINDOWS_LINE_ENDINGS := _FORMATTING_RULES,
            ],
        )
    )

    return "\n".join(filter(None, lines)).rstrip() + "\n"


# --------------------------------------------------------------------------
#   Parse LLM output results
# --------------------------------------------------------------------------
class ParsedCode(NamedTuple):
    """Container for parsed code components returned by an LLM."""
    function_imports: List[str]
    test_imports: List[str]
    main_function: str
    test_functions: Dict[str, str]
    main_function_name: str
    all_imports: List[str]          # De-duplicated + order-preserved


class ValidationError(Exception):
    """Raised when the LLM’s JSON does not satisfy the expected contract."""
    pass


# --------------------------------------------------------------------------- #
#                             Helper utilities                                #
# --------------------------------------------------------------------------- #
_STR_LIST_ERR = "`{field}` must be a list of strings (got {value!r})"


def _ensure_str_list(value: Any, field: str) -> List[str]:
    """Validate that *value* is a list of str; return a shallow copy."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValidationError(_STR_LIST_ERR.format(field=field, value=value))
    return list(value)


def _dedupe_preserve_order(seq: List[str]) -> List[str]:
    """Return *seq* without duplicates while preserving first-seen order."""
    seen = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


_IMPORT_RE = re.compile(r'^\s*(import|from)\s+[^\s]+')


def _strip_inline_imports(code: str) -> str:
    """
    Remove *top-level* import statements from a block of Python code.

    Parameters
    ----------
    code : str
        Raw code string as returned by the LLM.

    Returns
    -------
    str
        Same code but with any leading-level lines that start with
        `import …` or `from … import …` deleted.  Nested imports inside
        functions are left untouched.
    """
    kept_lines = [
        line for line in code.splitlines()
        if not _IMPORT_RE.match(line)
    ]
    return "\n".join(kept_lines)


# --------------------------------------------------------------------------- #
#                              Public API                                     #
# --------------------------------------------------------------------------- #
def parse_llm_json_output(json_data: Union[str, Dict[str, Any], Path]) -> ParsedCode:
    """
    Parse the JSON returned by an LLM and extract imports + code strings.

    (docstring unchanged for brevity)
    """
    # ── 1. Load JSON into a Python dict ───────────────────────────────────
    if isinstance(json_data, dict):
        data = json_data
    elif isinstance(json_data, (str, Path)):
        path = Path(json_data)
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            # Treat as raw JSON string
            data = json.loads(str(json_data))
    else:
        raise ValueError(
            "json_data must be a JSON string, dict, or filesystem path "
            f"(got type {type(json_data)})"
        )

    if not isinstance(data, dict):
        raise ValidationError("Top-level JSON must be an object")

    # ── 2. Validate + normalise import arrays ─────────────────────────────
    function_imports = _ensure_str_list(data.get("function_imports"), "function_imports")
    test_imports = _ensure_str_list(data.get("test_imports"),     "test_imports")

    # ── 3. Locate main + test functions ----------------------------------
    main_function_name = None
    main_function = None
    test_functions: Dict[str, str] = {}

    for key, val in data.items():
        # Skip non-code keys handled above
        if key in {"function_imports", "test_imports"}:
            continue

        if not isinstance(val, str):
            raise ValidationError(f"Value of key '{key}' must be a string containing code")

        if re.fullmatch(r"test_.*", key):           # canonical pytest test name
            test_functions[key] = val
        elif main_function_name is None:
            # First non-test key becomes the main function
            main_function_name, main_function = key, val
        else:
            # Multiple non-test code fields → ambiguous / unexpected
            raise ValidationError(
                "Multiple top-level code fields detected: "
                f"'{main_function_name}' and '{key}'. "
                "Exactly one main function is expected."
            )

    if main_function_name is None or main_function is None:
        raise ValidationError("No main function found in LLM JSON output")

    # ── NEW:  Remove any inline import statements from *main_function* ───
    import_re = re.compile(r'^\s*(import|from)\s+[^\s]+')
    main_function = "\n".join(
        line for line in main_function.splitlines()
        if not import_re.match(line)
    )

    # ── 4. Consolidate import lists, preserving order + de-duping ────────
    all_imports = _dedupe_preserve_order(function_imports + test_imports)

    return ParsedCode(
        function_imports=function_imports,
        test_imports=test_imports,
        main_function=main_function,
        test_functions=test_functions,
        main_function_name=main_function_name,
        all_imports=all_imports,
    )


def extract_function_signature(function_code: str) -> Dict[str, Union[str, List[str]]]:
    """
    Extract function signature information from function code.
    
    Args:
        function_code: String containing function definition
        
    Returns:
        Dict with function_name, parameters, parameter_types, return_type
        
    Raises:
        ValidationError: If function signature cannot be parsed
    """
    try:
        # Parse the function code into AST
        tree = ast.parse(function_code)
        
        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break
        
        if not func_def:
            raise ValidationError("No function definition found in code")
        
        # Extract function name
        function_name = func_def.name
        
        # Extract parameters and their types
        parameters = []
        parameter_types = []
        
        for arg in func_def.args.args:
            parameters.append(arg.arg)
            
            # Extract type annotation if present
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    parameter_types.append(arg.annotation.id)
                elif isinstance(arg.annotation, ast.Constant):
                    parameter_types.append(str(arg.annotation.value))
                else:
                    # For complex types, convert back to string
                    parameter_types.append(ast.unparse(arg.annotation))
            else:
                parameter_types.append("Any")
        
        # Extract return type annotation
        return_type = "Any"
        if func_def.returns:
            if isinstance(func_def.returns, ast.Name):
                return_type = func_def.returns.id
            elif isinstance(func_def.returns, ast.Constant):
                return_type = str(func_def.returns.value)
            else:
                return_type = ast.unparse(func_def.returns)
        
        return {
            'function_name': function_name,
            'parameters': parameters,
            'parameter_types': parameter_types,
            'return_type': return_type
        }
        
    except SyntaxError as e:
        raise ValidationError(f"Syntax error in function code: {e}")
    except Exception as e:
        raise ValidationError(f"Error parsing function signature: {e}")


def validate_function_signature(parsed_code: ParsedCode,
                                expected_signature: Dict) -> List[str]:
    """
    Validate that the parsed function matches the expected signature.

    Args:
        parsed_code: ParsedCode object from parse_llm_json_output
        expected_signature: Dict with expected function details
            - function_name: str
            - parameters: List[str]
            - parameter_types: List[str]
            - return_type: str

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: List[str] = []

    try:
        actual_sig = extract_function_signature(parsed_code.main_function)

        # ------------------------------------------------ function name -----
        expected_name = expected_signature.get(
            "function_name", parsed_code.main_function_name
        )
        if actual_sig["function_name"] != expected_name:
            errors.append(
                f"Function name mismatch: expected '{expected_name}', "
                f"got '{actual_sig['function_name']}'"
            )

        # ------------------------------------------------ param count -------
        expected_params = expected_signature.get("parameters", [])
        actual_params = actual_sig["parameters"]
        if len(actual_params) != len(expected_params):
            errors.append(
                f"Parameter count mismatch: expected {len(expected_params)}, "
                f"got {len(actual_params)}"
            )

        # ---------------------------------------------- param names ---------
        if len(actual_params) == len(expected_params):
            for i, (e_p, a_p) in enumerate(zip(expected_params, actual_params)):
                if e_p != a_p:
                    errors.append(
                        f"Parameter {i+1} name mismatch: expected '{e_p}', "
                        f"got '{a_p}'"
                    )

        # ---------------------------------------------- param types ---------
        expected_param_types = expected_signature.get("parameter_types", [])
        actual_param_types = actual_sig["parameter_types"]
        if expected_param_types and len(expected_param_types) == len(actual_param_types):
            for i, (e_t, a_t) in enumerate(zip(expected_param_types,
                                               actual_param_types)):
                e_norm, a_norm = _norm_type(e_t), _norm_type(a_t)
                if e_norm != "any" and a_norm != "any" and e_norm != a_norm:
                    errors.append(
                        f"Parameter {i+1} type mismatch: expected '{e_t}', "
                        f"got '{a_t}'"
                    )

        # ------------------------------------------------ return type -------
        expected_return = _norm_type(expected_signature.get("return_type"))
        actual_return   = _norm_type(actual_sig["return_type"])
        if expected_return != "any" and actual_return != "any" \
           and expected_return != actual_return:
            errors.append(
                f"Return type mismatch: expected '{expected_return}', "
                f"got '{actual_return}'"
            )

    except ValidationError as exc:
        errors.append(f"Signature parsing error: {exc}")

    return errors


def validate_basic_requirements(parsed_code: ParsedCode) -> List[str]:
    """
    Validate basic requirements for the parsed code.
    
    Args:
        parsed_code: ParsedCode object from parse_llm_json_output
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check that main function exists and is not empty
    if not parsed_code.main_function.strip():
        errors.append("Main function is empty")
    
    # Check that function code is syntactically valid Python
    try:
        ast.parse(parsed_code.main_function)
    except SyntaxError as e:
        errors.append(f"Main function has syntax error: {e}")
    
    # Check that function imports are valid Python identifiers/module names
    for imp in parsed_code.function_imports:
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$', imp):
            errors.append(f"Invalid import name: '{imp}'")
    
    # Check test functions for syntax if they exist
    for test_name, test_code in parsed_code.test_functions.items():
        if not test_code.strip():
            errors.append(f"Test function '{test_name}' is empty")
            continue
            
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            errors.append(f"Test function '{test_name}' has syntax error: {e}")
    
    # Check test imports
    for imp in parsed_code.test_imports:
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$', imp):
            errors.append(f"Invalid test import name: '{imp}'")
    
    return errors


def validate_against_task_signature(parsed_code: ParsedCode, task) -> List[str]:
    """
    Validate parsed code against task signature requirements.
    
    Args:
        parsed_code: ParsedCode object from parse_llm_json_output
        task: Task object with function_signature and expected_function_name
        
    Returns:
        List of validation error messages (empty if valid)
    """
    # Build expected signature from task
    sig = task.function_signature
    expected_signature = {
        'function_name': task.expected_function_name,
        'parameters': sig.parameters,
        'parameter_types': sig.parameter_types,
        'return_type': sig.return_shape
    }
    
    return validate_function_signature(parsed_code, expected_signature)


# Convenience function for complete validation
def validate_llm_output(json_data: Union[str, Dict, Path], task=None) -> tuple[ParsedCode, List[str]]:
    """
    Parse and validate LLM JSON output.
    
    Args:
        json_data: JSON string, dict, or file path
        task: Optional task object for signature validation
        
    Returns:
        Tuple of (ParsedCode object, list of validation errors)
    """
    try:
        parsed_code = parse_llm_json_output(json_data)
    except (json.JSONDecodeError, ValidationError) as e:
        return None, [f"Parsing error: {e}"]
    
    # Run basic validation
    errors = validate_basic_requirements(parsed_code)
    
    # Run task-specific validation if task provided
    if task:
        task_errors = validate_against_task_signature(parsed_code, task)
        errors.extend(task_errors)
    
    return parsed_code, errors


