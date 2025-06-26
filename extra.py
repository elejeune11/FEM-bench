

def build_prompt(task: Task, environment: Environment, dependency_code: str = "") -> str:
    """
    Build an LLM prompt from a task and environment specification.
    
    Args:
        task: Task specification
        environment: Environment configuration
        dependency_code: Code from previous tasks that this task depends on
        
    Returns:
        Formatted prompt string ready for LLM
    """
    sections = []
    
    # Header
    sections.append(f"# {task.title}")
    sections.append(f"**Task ID:** {task.task_id}")
    sections.append("")
    
    # Dependencies (if any)
    if task.task_dependencies or dependency_code:
        sections.append("## Dependencies")
        if task.task_dependencies:
            sections.append("This task requires the following functions from previous tasks:")
            for dep in task.task_dependencies:
                sections.append(f"- `{dep.name}` from task `{dep.source_task}`")
            sections.append("")
        
        if dependency_code:
            sections.append("**Required code from previous tasks:**")
            sections.append("```python")
            sections.append(dependency_code.strip())
            sections.append("```")
            sections.append("")
            sections.append("*Note: The above functions are already available in your environment.*")
            sections.append("")
    
    # Main task description
    sections.append("## Task Description")
    sections.append(task.prompt.strip())
    sections.append("")
    
    # Function specification
    sections.append("## Function Specification")
    sections.append(f"**Function name:** `{task.expected_function_name}`")
    sections.append("")
    sections.append("**Parameters:**")
    for param, param_type in zip(task.function_signature.parameters, task.function_signature.parameter_types):
        sections.append(f"- `{param}` ({param_type})")
    sections.append("")
    sections.append(f"**Returns:** {task.function_signature.return_shape}")
    sections.append("")
    
    # Environment
    sections.append("## Environment")
    sections.append(f"- Python {environment.python_version}")
    sections.append("- Required imports: " + ", ".join(f"`{imp}`" for imp in environment.testing.get('required_imports', [])))
    if environment.testing.get('allowed_imports'):
        sections.append("- Additional imports available: " + ", ".join(f"`{imp}`" for imp in environment.testing.get('allowed_imports', [])))
    sections.append("")
    
    # Testing requirements
    if task.include_tests:
        sections.append("## Testing Requirements")
        sections.append(f"- Framework: {environment.testing.get('framework', 'pytest')}")
        if task.expected_test_functions:
            sections.append("- Required test functions: " + ", ".join(f"`{func}()`" for func in task.expected_test_functions))
        sections.append("")
    
    # Response format - this is the key part for structured output
    sections.append("## Response Format")
    sections.append("Please provide your solution in the following structured format:")
    sections.append("")
    sections.append("```")
    sections.append("<solution>")
    sections.append("<imports>")
    sections.append("# All necessary imports here")
    sections.append("</imports>")
    sections.append("")
    sections.append("<function>")
    sections.append(f"def {task.expected_function_name}({', '.join(task.function_signature.parameters)}):")
    sections.append('    """Your docstring here."""')
    sections.append("    # Your implementation here")
    sections.append("    pass")
    sections.append("</function>")
    sections.append("")
    if task.include_tests:
        sections.append("<tests>")
        for test_func in task.expected_test_functions:
            sections.append(f"def {test_func}():")
            sections.append('    """Test description."""')
            sections.append("    # Your test implementation here")
            sections.append("    pass")
            sections.append("")
        sections.append("</tests>")
    sections.append("</solution>")
    sections.append("```")
    sections.append("")
    
    # Final instructions
    sections.append("## Instructions")
    sections.append("1. Implement the function according to the specifications above")
    if task.include_tests:
        sections.append("2. Include comprehensive unit tests that verify correctness")
    sections.append("3. Follow the exact response format with XML tags")
    sections.append("4. Ensure all code is properly indented and syntactically correct")
    
    return "\n".join(sections)


def parse_structured_response(response: str) -> Dict[str, str]:
    """
    Parse a structured XML response from an LLM.
    
    Args:
        response: The LLM response containing XML tags
        
    Returns:
        Dictionary with 'imports', 'function', and 'tests' code sections
        
    Raises:
        ValueError: If required sections are missing or malformed
    """
    import re
    
    result = {}
    
    # Extract imports section
    imports_match = re.search(r'<imports>(.*?)</imports>', response, re.DOTALL)
    if imports_match:
        result['imports'] = imports_match.group(1).strip()
    else:
        result['imports'] = ""
    
    # Extract function section
    function_match = re.search(r'<function>(.*?)</function>', response, re.DOTALL)
    if function_match:
        result['function'] = function_match.group(1).strip()
    else:
        raise ValueError("No <function> section found in response")
    
    # Extract tests section
    tests_match = re.search(r'<tests>(.*?)</tests>', response, re.DOTALL)
    if tests_match:
        result['tests'] = tests_match.group(1).strip()
    else:
        result['tests'] = ""
    
    return result


def combine_code_sections(parsed_response: Dict[str, str]) -> str:
    """
    Combine parsed response sections into executable Python code.
    
    Args:
        parsed_response: Output from parse_structured_response()
        
    Returns:
        Complete Python code ready for execution
    """
    sections = []
    
    # Add imports if present
    if parsed_response.get('imports'):
        sections.append(parsed_response['imports'])
        sections.append("")  # Blank line
    
    # Add main function
    if parsed_response.get('function'):
        sections.append(parsed_response['function'])
        sections.append("")  # Blank line
    
    # Add test functions
    if parsed_response.get('tests'):
        sections.append(parsed_response['tests'])
    
    return "\n".join(sections).strip()


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


def build_dependency_code(task: Task, completed_tasks: Dict[str, str]) -> str:
    """
    Build dependency code string from completed tasks.
    
    Args:
        task: Current task that may have dependencies
        completed_tasks: Dictionary mapping task_id -> completed code
        
    Returns:
        Combined dependency code, or empty string if no dependencies
        
    Raises:
        ValueError: If required dependency is missing
    """
    if not task.task_dependencies:
        return ""
    
    dependency_sections = []
    
    for dep in task.task_dependencies:
        if dep.source_task not in completed_tasks:
            raise ValueError(f"Missing dependency: {dep.source_task} required for {dep.name}")
        
        # Extract the specific function from the completed task
        source_code = completed_tasks[dep.source_task]
        
        # For now, include the entire source code
        # You might want to extract just the specific function later
        dependency_sections.append(f"# From task {dep.source_task}")
        dependency_sections.append(source_code.strip())
        dependency_sections.append("")
    
    return "\n".join(dependency_sections).strip()


def extract_function_from_code(code: str, function_name: str) -> str:
    """
    Extract a specific function definition from code.
    
    Args:
        code: Python code containing multiple functions
        function_name: Name of function to extract
        
    Returns:
        Just the function definition, or empty string if not found
    """
    import re
    import ast
    
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Get the function's source lines
                lines = code.split('\n')
                start_line = node.lineno - 1  # AST is 1-indexed
                
                # Find the end of the function (next function or end of file)
                end_line = len(lines)
                for next_node in ast.walk(tree):
                    if (isinstance(next_node, ast.FunctionDef) and 
                        next_node.lineno > node.lineno):
                        end_line = min(end_line, next_node.lineno - 1)
                
                # Extract function lines
                function_lines = lines[start_line:end_line]
                
                # Remove trailing empty lines
                while function_lines and not function_lines[-1].strip():
                    function_lines.pop()
                
                return '\n'.join(function_lines)
    
    except (SyntaxError, ValueError):
        # If parsing fails, return empty string
        pass
    
    return ""
