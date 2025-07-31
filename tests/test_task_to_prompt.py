from fem_bench.task_base import Task
from fem_bench.task_to_prompt import extract_test_name_and_docstring
from fem_bench.task_to_prompt import extract_signature_and_docstring
from fem_bench.task_to_prompt import task_to_code_prompt, task_to_test_prompt
import pytest
import textwrap


def test_task_to_code_prompt_includes_signature_docstring_and_helper():
    # --- Main function source ---
    main_fcn_code = textwrap.dedent("""
    def foo(a: int, b: int) -> int:
        \"\"\"
        Add two integers.

        Parameters:
            a (int): The first number.
            b (int): The second number.

        Returns:
            int: The sum of a and b.
        \"\"\"
        return add(a, b)
    """)

    # --- Helper function source ---
    helper_code = textwrap.dedent("""
    def add(x: int, y: int) -> int:
        \"\"\"
        Return the sum of two integers.

        Parameters:
            x (int): First number.
            y (int): Second number.

        Returns:
            int: The result of x + y.
        \"\"\"
        return x + y
    """)

    # --- Task construction ---
    task = Task(
        task_id="foo_task",
        task_short_description="adds two integers using a helper",
        created_date="2025-07-09",
        created_by="test_user",
        main_fcn_code=main_fcn_code,
        required_imports=[],
        fcn_dependency_code=[helper_code],
        reference_verification_inputs=[[1, 2]],
        test_cases=[]
    )

    # --- Generate the prompt ---
    prompt = task_to_code_prompt(task)

    # --- Assertions ---
    # Instructional content
    assert "# Python Function Implementation Task" in prompt
    assert "## Requirements:" in prompt
    assert "- Output only valid Python code" in prompt
    assert "## Function Signature:" in prompt

    # Main function presence
    assert "def foo(a: int, b: int) -> int" in prompt
    assert "Add two integers." in prompt
    assert "int: The sum of a and b." in prompt

    # Helper function presence
    assert "## Available Helper Functions" in prompt
    assert "def add(x: int, y: int) -> int" in prompt
    assert "Return the sum of two integers." in prompt
    assert "int: The result of x + y." in prompt


def test_task_to_test_prompt_includes_function_and_tests():
    # --- Main function code ---
    main_fcn_code = textwrap.dedent("""
    def linear_uniform_mesh_1D(x_min: float, x_max: float, num_elements: int) -> np.ndarray:
        \"\"\"
        Generate a 1D linear mesh.

        Parameters:
            x_min (float): Start of the domain.
            x_max (float): End of the domain.
            num_elements (int): Number of elements.

        Returns:
            np.ndarray[np.ndarray, np.ndarray]: Node coordinates and element connectivity.
        \"\"\"
        pass
    """)

    # --- Reference test code ---
    test_code = textwrap.dedent("""
    def test_basic_mesh_creation(fcn):
        \"\"\"Test basic mesh creation with simple parameters.\"\"\"
        pass
    """)

    task = Task(
        task_id="mesh_task",
        task_short_description="creates a uniform 1D mesh",
        created_date="2025-07-09",
        created_by="test_user",
        main_fcn_code=main_fcn_code,
        required_imports=["import numpy as np"],
        fcn_dependency_code=[],
        reference_verification_inputs=[[0.0, 1.0, 4]],
        test_cases=[
            {"test_code": test_code, "expected_failures": []}
        ]
    )

    prompt = task_to_test_prompt(task)

    # --- Assertions ---
    assert "def linear_uniform_mesh_1D" in prompt
    assert "Generate a 1D linear mesh." in prompt
    assert "test_basic_mesh_creation" in prompt
    assert "Test basic mesh creation with simple parameters." in prompt
    assert "Output only valid Python code" in prompt
    assert "Use the exact test function names listed below" in prompt


def test_extract_name_and_doc_with_docstring():
    code = '''
def test_addition():
    """Tests that addition works."""
    assert 1 + 1 == 2
'''
    name, doc = extract_test_name_and_docstring(code)
    assert name == "test_addition"
    assert doc == "Tests that addition works."


def test_extract_name_and_doc_without_docstring():
    code = '''
def test_subtraction():
    assert 2 - 1 == 1
'''
    name, doc = extract_test_name_and_docstring(code)
    assert name == "test_subtraction"
    assert doc == ""


def test_extract_name_and_doc_no_function():
    code = '''
x = 42
y = x + 1
'''
    with pytest.raises(ValueError, match="No test function found."):
        extract_test_name_and_docstring(code)


def test_signature_and_docstring_with_annotations():
    code = '''
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b
'''
    sig, doc = extract_signature_and_docstring(code)
    assert sig == "def add(a: int, b: int) -> int:"
    assert doc == textwrap.indent('"""\nAdd two integers and return the sum.\n"""', "    ")


def test_signature_without_return_annotation():
    code = '''
def greet(name: str):
    """Return a greeting for the user."""
    return f"Hello, {name}"
'''
    sig, doc = extract_signature_and_docstring(code)
    assert sig == "def greet(name: str):"
    assert doc == textwrap.indent('"""\nReturn a greeting for the user.\n"""', "    ")


def test_signature_without_docstring():
    code = '''
def square(x: float) -> float:
    return x * x
'''
    sig, doc = extract_signature_and_docstring(code)
    assert sig == "def square(x: float) -> float:"
    assert doc == textwrap.indent('"""\n\n"""', "    ")  # Empty docstring format


def test_no_function_raises_value_error():
    code = '''
x = 5
y = x + 2
'''
    with pytest.raises(ValueError, match="No function definition found."):
        extract_signature_and_docstring(code)


def test_task_to_test_prompt_with_exceptions():
    broken_main_code = "def broken(:)"  # Invalid syntax for signature extraction
    broken_test_code = {"test_code": "def broken_test(:"}  # Also invalid syntax

    task = Task(
        task_id="test001",
        task_short_description="Broken task",
        created_date="2025-07-30",
        created_by="Tester",
        main_fcn_code=broken_main_code,
        test_cases=[broken_test_code]
    )

    prompt = task_to_test_prompt(task)

    assert "def <unknown>():" in prompt
    assert '"""Missing docstring."""' in prompt
    assert "- (no test cases found)" in prompt