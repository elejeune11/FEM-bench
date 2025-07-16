import textwrap
from fem_bench.task_base import Task
from fem_bench.task_to_prompt import task_to_code_prompt, task_to_test_prompt


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
