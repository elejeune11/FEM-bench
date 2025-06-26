from fem_bench.yaml_load import Task, TaskDependency, FunctionSignature
from fem_bench.yaml_load import Task, FunctionSignature, Parameter, CodeBlock

from fem_bench.yaml_load import load_task, load_environment
from fem_bench.prompt import build_dependency_code, estimate_token_count
# from fem_bench.prompt import parse_llm_json_output, validate_function_signature, validate_basic_requirements, ParsedCode, ValidationError
# from fem_bench.prompt import validate_against_task_signature, validate_llm_output
from fem_bench.prompt import build_prompt

import json
import pytest
from pathlib import Path
import re

# Get the test files directory
TEST_FILES_DIR = Path(__file__).parent / "files"


def test_estimate_token_count():
    """Test token count estimation."""
    short_text = "Hello world"
    long_text = "a" * 1000
    
    short_tokens = estimate_token_count(short_text)
    long_tokens = estimate_token_count(long_text)
    
    assert short_tokens == len(short_text) // 4
    assert long_tokens == 250  # 1000 / 4
    assert long_tokens > short_tokens


def test_estimate_token_count_empty():
    """Test token estimation for empty string."""
    assert estimate_token_count("") == 0


def test_build_dependency_code_no_dependencies():
    """Test that a task with no dependencies returns empty string."""
    # Load T1_SF_001.yaml which has no dependencies
    task = load_task(TEST_FILES_DIR / "T1_SF_001.yaml")
    
    result = build_dependency_code(task, task_dir=TEST_FILES_DIR)
    assert result == ""


def test_build_dependency_code_one_dependency():
    """Test that a task with one dependency loads and formats correctly."""
    task = load_task(TEST_FILES_DIR / "T1_SF_002.yaml")
    result = build_dependency_code(task, task_dir=TEST_FILES_DIR)

    assert len(result) > 0
    assert "compute_1d_linear_shape_functions" in result
    assert "T1_SF_001" in result
    assert "Description:" in result
    assert "Signature:" in result
    assert "def " in result
    assert "->" in result
    assert "xi: float" in result
    assert "-> numpy.ndarray" in result  # this is the correct return type


def test_build_dependency_code_two_dependencies():
    """Test that a task with two dependencies loads and formats both correctly."""
    # Load T1_EX_001.yaml which depends on both T1_SF_001 and T1_SF_002
    task = load_task(TEST_FILES_DIR / "T1_EX_001.yaml")
    
    result = build_dependency_code(task, task_dir=TEST_FILES_DIR)
    
    # Check that we get output
    assert len(result) > 0
    
    # Check for first dependency (linear shape functions)
    assert "compute_1d_linear_shape_functions" in result
    assert "T1_SF_001" in result
    
    # Check for second dependency (quadratic shape functions)
    assert "compute_1d_quadratic_shape_functions" in result
    assert "T1_SF_002" in result
    
    # Check general formatting
    assert "Description:" in result
    assert "Signature:" in result
    assert result.count("def ") >= 2  # Should have at least 2 function signatures
    assert result.count("->") >= 2    # Should have at least 2 return types
    
    # Check specific descriptions
    assert "Compute linear shape functions for 1D two-node element" in result
    assert "Compute quadratic shape functions for 1D three-node element" in result


def test_build_dependency_code_output_format():
    """Test the exact format of the dependency output."""
    # Load task that depends on T1_SF_001
    task = load_task(TEST_FILES_DIR / "T1_SF_002.yaml")
    
    result = build_dependency_code(task, task_dir=TEST_FILES_DIR)

    # Expected structure in the output
    expected_parts = [
        "**compute_1d_linear_shape_functions** (from T1_SF_001):",
        "- Description: Compute linear shape functions for 1D two-node element",
        "- Signature: `def compute_1d_linear_shape_functions(xi: float) -> numpy.ndarray`"
    ]

    for part in expected_parts:
        assert part in result, f"Missing expected part in output:\n{part}\n\nFull result:\n{result}"


def test_build_dependency_code_missing_file():
    """Test error handling when a dependency file is missing."""
    
    task = Task(
        task_id="T1_TEST_MISSING",
        category="test",
        subcategory="test",
        title="Test Missing Dependency",
        short_description="Test task with missing dependency",
        version="1.0",
        created_date="2025-06-04",
        created_by="test",
        prompt_description="Test prompt",
        expected_function_name="test_function",
        include_tests=False,
        expected_test_functions=[],
        function_signature=FunctionSignature(
            input_parameters=[Parameter(name="x", type="int")],
            return_parameters=[Parameter(name="result", type="int")]
        ),
        task_dependencies={
            "required_functions": [
                TaskDependency(function_name="some_dependency", source_task="NONEXISTENT_TASK")
            ]
        },
        reference_solution=CodeBlock(code="def test_function(x): return x"),
        failure_examples={},
        reference_verification={"test_cases": []},
        test_efficacy_verification={"expected_failures": []}
    )

    with pytest.raises(FileNotFoundError):
        build_dependency_code(task, task_dir=TEST_FILES_DIR)


def test_yaml_files_load_correctly():
    """Test that all our YAML files can be loaded successfully and validate core fields."""
    # T1_SF_001: no dependencies
    task1 = load_task(TEST_FILES_DIR / "T1_SF_001.yaml")
    assert task1.task_id == "T1_SF_001"
    assert task1.expected_function_name == "compute_1d_linear_shape_functions"
    assert task1.task_dependencies.get("required_functions", []) == []

    # T1_SF_002: one dependency
    task2 = load_task(TEST_FILES_DIR / "T1_SF_002.yaml")
    assert task2.task_id == "T1_SF_002"
    assert task2.expected_function_name == "compute_1d_quadratic_shape_functions"
    deps2 = task2.task_dependencies.get("required_functions", [])
    assert len(deps2) == 1
    assert deps2[0].source_task == "T1_SF_001"

    # T1_EX_001: two dependencies
    task3 = load_task(TEST_FILES_DIR / "T1_EX_001.yaml")
    assert task3.task_id == "T1_EX_001"
    assert task3.expected_function_name == "analyze_shape_function_set"
    deps3 = task3.task_dependencies.get("required_functions", [])
    assert len(deps3) == 2
    sources = [d.source_task for d in deps3]
    assert "T1_SF_001" in sources
    assert "T1_SF_002" in sources


def test_dependency_chain():
    """Test that dependencies work in a chain scenario."""
    # Load all tasks in the chain
    task1 = load_task(TEST_FILES_DIR / "T1_SF_001.yaml")  # No dependencies
    task2 = load_task(TEST_FILES_DIR / "T1_SF_002.yaml")  # Depends on T1_SF_001
    task3 = load_task(TEST_FILES_DIR / "T1_EX_001.yaml")  # Depends on T1_SF_001 + T1_SF_002

    # Verify dependency counts
    deps1 = task1.task_dependencies.get("required_functions", [])
    deps2 = task2.task_dependencies.get("required_functions", [])
    deps3 = task3.task_dependencies.get("required_functions", [])

    assert len(deps1) == 0
    assert len(deps2) == 1
    assert len(deps3) == 2

    # Generate dependency code
    result1 = build_dependency_code(task1, task_dir=TEST_FILES_DIR)
    result2 = build_dependency_code(task2, task_dir=TEST_FILES_DIR)
    result3 = build_dependency_code(task3, task_dir=TEST_FILES_DIR)

    assert result1 == ""  # No dependencies
    assert len(result2) > 0  # One dependency
    assert len(result3) > len(result2)  # Two dependencies → longer output


def _make_prompt():
    """Utility: load a simple task & env, return the generated prompt."""
    task = load_task(TEST_FILES_DIR / "T1_SF_001.yaml")
    env = load_environment(TEST_FILES_DIR / "tier1_environment_example.yaml")
    return task, env, build_prompt(task, env, task_dir=TEST_FILES_DIR)


def test_build_prompt_no_dependencies():
    task, env, prompt = _make_prompt()

    # ── Environment block ────────────────────────────────────────────────
    assert "# Environment Configuration" in prompt
    assert env.environment_name in prompt
    assert f"(Tier {env.tier})" in prompt
    assert env.description in prompt

    # ── Library sections ────────────────────────────────────────────────
    assert "Required Libraries" in prompt
    for lib in env.required_libraries:
        assert lib.name in prompt
        assert lib.version in prompt
        if lib.import_as:
            assert lib.import_as in prompt

    if env.allowed_libraries:
        assert "Allowed Libraries" in prompt
        for lib in env.allowed_libraries:
            assert lib.name in prompt

    # ── Task specification ──────────────────────────────────────────────
    assert "# Task Specification" in prompt
    assert task.title in prompt
    assert task.short_description in prompt
    assert task.expected_function_name in prompt
    assert f"def {task.expected_function_name}" in prompt

    # ── No “Task Dependencies” section for dependency-free tasks ───────
    assert "# Task Dependencies" not in prompt

    # ── Output-format rules present ─────────────────────────────────────
    assert "# Output Format Requirements" in prompt
    assert "Respond with valid JSON" in prompt
    assert "function_imports" in prompt
    if task.include_tests and task.expected_test_functions:
        assert "test_imports" in prompt
        for test_func in task.expected_test_functions:
            assert test_func in prompt



def test_build_prompt_with_one_dependency():
    """Test build_prompt with a task that has one dependency."""
    # Load task with one dependency
    task = load_task(TEST_FILES_DIR / "T1_SF_002.yaml")
    environment = load_environment(TEST_FILES_DIR / "tier1_environment_example.yaml")
    
    prompt = build_prompt(task, environment, task_dir=TEST_FILES_DIR)
    
    # Check that dependencies section is included
    assert "Task Dependencies" in prompt
    assert "compute_1d_linear_shape_functions" in prompt
    assert "T1_SF_001" in prompt
    
    # Check task-specific content
    assert "T1_SF_002" not in prompt  # Task ID shouldn't be in prompt
    assert "compute_1d_quadratic_shape_functions" in prompt
    assert task.title in prompt
    

def test_build_prompt_with_multiple_dependencies():
    """Test build_prompt with a task that has multiple dependencies."""
    # Load task with multiple dependencies
    task = load_task(TEST_FILES_DIR / "T1_EX_001.yaml")
    environment = load_environment(TEST_FILES_DIR / "tier1_environment_example.yaml")
    
    prompt = build_prompt(task, environment, task_dir=TEST_FILES_DIR)

    # ── 1. Dependency block ─────────────────────────────────────────────
    assert "Task Dependencies" in prompt
    
    # Check that dependency function names and source task IDs appear
    dependency_functions = [dep.function_name for dep in task.task_dependencies.get("required_functions", [])]
    dependency_sources = [dep.source_task for dep in task.task_dependencies.get("required_functions", [])]

    for func in dependency_functions:
        assert func in prompt, f"Missing dependency function `{func}` in prompt"

    for source in dependency_sources:
        assert source in prompt, f"Missing source task ID `{source}` in prompt"

    # ── 2. Task-specific content ────────────────────────────────────────
    assert task.title in prompt
    assert task.short_description in prompt
    assert task.expected_function_name in prompt
    assert f"def {task.expected_function_name}" in prompt

    # ── 3. Output format requirements ───────────────────────────────────
    assert "# Output Format Requirements" in prompt
    assert "Respond with valid JSON" in prompt


# ---------------------------------------------------------------------------
# Environment-section verification (rev. 2025-06-20)
# ---------------------------------------------------------------------------
def test_build_prompt_environment_sections():
    _, env, prompt = _make_prompt()

    # ── Core environment info always present ────────────────────────────
    assert "# Environment Configuration" in prompt
    assert "Required Libraries" in prompt
    if env.allowed_libraries:
        assert "Allowed Libraries" in prompt

    # ── Import Guidelines ───────────────────────────────────────────────
    if env.import_guidelines:
        assert "Import Guidelines" in prompt
        assert "import numpy as np" in prompt

    # ── Code Requirements ───────────────────────────────────────────────
    if env.code_requirements:
        assert "Code Requirements" in prompt
        for key, value in env.code_requirements.model_dump().items():
            assert str(value) in prompt


def test_build_prompt_test_functions():
    task, _, prompt = _make_prompt()

    if task.include_tests and task.expected_test_functions:
        # No dedicated “Required Test Functions” header anymore.
        for test_func in task.expected_test_functions:
            # Look for the test-function name anywhere in the prompt
            # (should be inside the JSON example).
            assert re.search(rf"\b{re.escape(test_func)}\b", prompt), (
                f"Test function '{test_func}' missing from prompt"
            )


def test_build_prompt_structure():
    """Test the overall structure and formatting of the prompt."""
    task = load_task(TEST_FILES_DIR / "T1_SF_002.yaml")
    environment = load_environment(TEST_FILES_DIR / "tier1_environment_example.yaml")
    
    prompt = build_prompt(task, environment, task_dir=TEST_FILES_DIR)
    
    # Check major sections are in correct order
    sections = [
        "# Environment Configuration",
        "## Required Libraries", 
        "# Task Dependencies",  # Should be present for T1_SF_002
        "# Task Specification",
        "## Required Function",
        "## Task Details",
        "# Implementation Instructions"
    ]
    
    last_position = -1
    for section in sections:
        if section in prompt:  # Some sections might not be present
            position = prompt.find(section)
            assert position > last_position, f"Section '{section}' is out of order"
            last_position = position


def test_build_prompt_empty_dependencies():
    """Test that empty dependencies are handled gracefully."""
    task = load_task(TEST_FILES_DIR / "T1_SF_001.yaml")  # No dependencies
    environment = load_environment(TEST_FILES_DIR / "tier1_environment_example.yaml")
    
    prompt = build_prompt(task, environment, task_dir=TEST_FILES_DIR)
    
    # Should not have dependencies section
    assert "Task Dependencies" not in prompt
    assert "Your implementation can use the following functions" not in prompt


def test_build_prompt_library_formatting():
    """Test that libraries are properly formatted in the prompt."""
    task = load_task(TEST_FILES_DIR / "T1_SF_001.yaml")
    environment = load_environment(TEST_FILES_DIR / "tier1_environment_example.yaml")
    
    prompt = build_prompt(task, environment, task_dir=TEST_FILES_DIR)
    
    # Check required libraries formatting
    for lib in environment.required_libraries:
        assert lib.name in prompt
        assert lib.purpose in prompt
        if lib.import_as:
            assert f"import as `{lib.import_as}`" in prompt
    
    # Check allowed libraries formatting
    for lib in environment.allowed_libraries:
        assert lib.name in prompt
        assert lib.purpose in prompt


def test_build_prompt_minimal_environment():
    """Test build_prompt with minimal environment data."""
    # Create a minimal task for testing
    task = load_task(TEST_FILES_DIR / "T1_SF_001.yaml")
    
    # Load environment (assuming it has all required fields)
    environment = load_environment(TEST_FILES_DIR / "tier1_environment_example.yaml")
    
    # Should not crash and should include basic information
    prompt = build_prompt(task, environment)
    
    assert len(prompt) > 0
    assert task.expected_function_name in prompt
    assert environment.environment_name in prompt


def test_build_prompt_returns_string():
    """Test that build_prompt returns a string."""
    task = load_task(TEST_FILES_DIR / "T1_SF_001.yaml")
    environment = load_environment(TEST_FILES_DIR / "tier1_environment_example.yaml")
    
    prompt = build_prompt(task, environment, task_dir=TEST_FILES_DIR)
    
    assert isinstance(prompt, str)
    assert len(prompt) > 100  # Should be a substantial prompt


# def test_build_prompt_advanced():
#     task = load_task(TEST_FILES_DIR / "T1_FE_008.yaml")
#     aa = 44


# --------------------------------------------------------------------------
#   Parse LLM output results
# --------------------------------------------------------------------------

# @pytest.fixture
# def test_files_dir():
#     """Get the test files directory."""
#     return Path(__file__).parent / "files"


# def test_parse_valid_file(test_files_dir):
#     """Test parsing the valid JSON file."""
#     result = parse_llm_json_output(test_files_dir / "valid.json")
    
#     assert isinstance(result, ParsedCode)
#     assert result.main_function_name
#     assert result.main_function


# def test_parse_invalid_file(test_files_dir):
#     """Test parsing the invalid JSON file."""
#     with pytest.raises((json.JSONDecodeError, ValidationError)):
#         parse_llm_json_output(test_files_dir / "invalid.json")


# def test_validate_signature_valid():
#     """Test validation with matching signature."""
#     parsed_code = ParsedCode(
#         function_imports=["math"],
#         test_imports=[],
#         main_function='def calculate_sum(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b',
#         test_functions={},
#         main_function_name="calculate_sum",
#         all_imports=["math"]
#     )
    
#     expected_signature = {
#         'function_name': 'calculate_sum',
#         'parameters': ['a', 'b'],
#         'parameter_types': ['int', 'int'],
#         'return_type': 'int'
#     }
    
#     errors = validate_function_signature(parsed_code, expected_signature)
#     assert errors == []


# def test_validate_signature_invalid():
#     """Test validation with mismatched signature."""
#     parsed_code = ParsedCode(
#         function_imports=[],
#         test_imports=[],
#         main_function='def wrong_name(x: str) -> float:\n    """Wrong function."""\n    return 1.0',
#         test_functions={},
#         main_function_name="wrong_name",
#         all_imports=[]
#     )
    
#     expected_signature = {
#         'function_name': 'calculate_sum',
#         'parameters': ['a', 'b'], 
#         'parameter_types': ['int', 'int'],
#         'return_type': 'int'
#     }
    
#     errors = validate_function_signature(parsed_code, expected_signature)
#     assert len(errors) > 0
#     assert any("Function name mismatch" in error for error in errors)


# def test_validate_basic_requirements_valid():
#     """Test validation with valid code."""
#     parsed_code = ParsedCode(
#         function_imports=["numpy", "math"],
#         test_imports=["pytest"],
#         main_function='def calculate_sum(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b',
#         test_functions={
#             "test_calculate_sum": 'def test_calculate_sum():\n    """Test the function."""\n    assert calculate_sum(2, 3) == 5'
#         },
#         main_function_name="calculate_sum",
#         all_imports=["numpy", "math", "pytest"]
#     )
    
#     errors = validate_basic_requirements(parsed_code)
#     assert errors == []


# def test_validate_basic_requirements_invalid():
#     """Test validation with invalid code."""
#     parsed_code = ParsedCode(
#         function_imports=["123invalid", "numpy-bad"],
#         test_imports=["pytest..bad"],
#         main_function='def broken_function(\n    # Missing closing paren and colon',
#         test_functions={
#             "test_broken": 'def test_broken(\n    # Syntax error in test too',
#             "test_empty": '   '  # Empty test
#         },
#         main_function_name="broken_function",
#         all_imports=["123invalid"]
#     )
    
#     errors = validate_basic_requirements(parsed_code)
#     assert len(errors) > 0
#     assert any("syntax error" in error.lower() for error in errors)
#     assert any("Invalid import name" in error for error in errors)
#     assert any("empty" in error for error in errors)


# def test_validate_against_task_signature_valid(test_files_dir):
#     """Test validation with matching task signature."""
#     task = load_task(test_files_dir / "test_001.yaml")
    
#     parsed_code = ParsedCode(
#         function_imports=["math"],
#         test_imports=["pytest"],
#         main_function='def calculate_sum(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b',
#         test_functions={},
#         main_function_name="calculate_sum",
#         all_imports=["math", "pytest"]
#     )
    
#     errors = validate_against_task_signature(parsed_code, task)
#     assert errors == []


# def test_validate_against_task_signature_invalid(test_files_dir):
#     """Test validation with mismatched task signature."""
#     task = load_task(test_files_dir / "test_001.yaml")
    
#     parsed_code = ParsedCode(
#         function_imports=[],
#         test_imports=[],
#         main_function='def wrong_name(x: str) -> float:\n    """Wrong function."""\n    return 1.0',
#         test_functions={},
#         main_function_name="wrong_name",
#         all_imports=[]
#     )
    
#     errors = validate_against_task_signature(parsed_code, task)
#     assert len(errors) > 0
#     assert any("Function name mismatch" in error for error in errors)


# def test_validate_llm_output_valid_with_task(test_files_dir):
#     """Test validation with valid JSON and matching task."""
#     task = load_task(test_files_dir / "test_001.yaml")
    
#     parsed_code, errors = validate_llm_output(
#         test_files_dir / "test_001_matching.json",
#         task=task
#     )
    
#     assert isinstance(parsed_code, ParsedCode)
#     assert errors == []
#     assert parsed_code.main_function_name == "calculate_sum"


# def test_validate_llm_output_invalid_with_task(test_files_dir):
#     """Test validation with invalid JSON or mismatched task."""
#     task = load_task(test_files_dir / "test_001.yaml")
    
#     parsed_code, errors = validate_llm_output(
#         test_files_dir / "test_001_mismatched.json",
#         task=task
#     )
    
#     # Should either fail to parse or have validation errors
#     if parsed_code is None:
#         assert len(errors) > 0
#         assert "Parsing error" in errors[0]
#     else:
#         assert len(errors) > 0


