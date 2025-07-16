import numpy as np
import pytest
from fem_bench.evaluate_output import evaluate_function_output_match
from fem_bench.evaluate_output import load_function_from_code
from fem_bench.evaluate_output import evaluate_task_tests, compile_function_from_string, run_test_case
from fem_bench.task_base import Task
import textwrap


# Reference function
def ref_add_and_scale(x, y):
    return (x + y, np.array([x, y]) * 2)


# Matching LLM-generated function
def gen_add_and_scale_correct(x, y):
    return (x + y, np.array([x, y]) * 2)


# Incorrect result
def gen_add_and_scale_wrong(x, y):
    return (x * y, np.array([x, y]) * 2)


# Raises an error
def gen_broken(x, y):
    raise RuntimeError("Simulated crash")


def test_evaluate_function_output_match_success():
    inputs = [[1, 2], [3, 4]]
    assert evaluate_function_output_match(ref_add_and_scale, gen_add_and_scale_correct, inputs)


def test_evaluate_function_output_match_mismatch():
    inputs = [[1, 2], [3, 4]]
    assert not evaluate_function_output_match(ref_add_and_scale, gen_add_and_scale_wrong, inputs)


def test_evaluate_function_output_match_exception():
    inputs = [[1, 2]]
    assert not evaluate_function_output_match(ref_add_and_scale, gen_broken, inputs)


def test_loads_valid_function_and_executes():
    code = """
def multiply(a: int, b: int) -> int:
    return a * b
"""
    fcn = load_function_from_code(code)
    assert callable(fcn)
    assert fcn.__name__ == "multiply"
    assert fcn(2, 3) == 6


def test_load_function_with_import():
    code = """
def my_sqrt(x):
    return np.sqrt(x)
"""
    imports = ["import numpy as np"]
    f = load_function_from_code(code, required_imports=imports)
    assert round(f(4), 5) == 2.0


def test_load_function_with_helper():
    helper = """
def double(x):
    return 2 * x
"""
    code = """
def quadruple(x):
    return double(double(x))
"""
    f = load_function_from_code(code, fcn_dependencies=[helper])
    assert f(2) == 8


def test_load_function_with_imports_and_helpers():
    imports = ["import numpy as np"]
    helpers = ["""
def square(x):
    return x * x
"""]
    code = """
def square_plus_one(x):
    return square(x) + np.ones(1)[0]
"""
    f = load_function_from_code(code, required_imports=imports, fcn_dependencies=helpers)
    assert f(3) == 10.0


def test_load_function_with_missing_function_raises():
    bad_code = "x = 2 + 2"
    with pytest.raises(ValueError, match="No function definition found"):
        load_function_from_code(bad_code)


def test_raises_error_if_no_function_found():
    code = "x = 42"
    with pytest.raises(ValueError, match="No function definition found in the code"):
        load_function_from_code(code)


def test_evaluate_task_tests_multiple_tests_and_failures():
    # --- Reference (correct) implementation ---
    ref_code = textwrap.dedent("""
    def add(a: int, b: int) -> int:
        return a + b
    """)

    # --- Two incorrect implementations ---
    fail_code_1 = textwrap.dedent("""
    def add(a: int, b: int) -> int:
        return a - b
    """)

    fail_code_2 = textwrap.dedent("""
    def add(a: int, b: int) -> int:
        return a * b
    """)

    # --- Test 1: basic positive addition ---
    test_code_1 = textwrap.dedent("""
    def test_add_positive(fcn):
        \"\"\"Test that adding 2 and 3 returns 5.\"\"\"
        assert fcn(2, 3) == 5
    """)

    # --- Test 2: zero edge case ---
    test_code_2 = textwrap.dedent("""
    def test_add_zero(fcn):
        \"\"\"Test that adding 0 and 5 returns 5.\"\"\"
        assert fcn(0, 5) == 5
    """)

    # --- Create Task with two test cases, each expecting both failures ---
    task = Task(
        task_id="add_task",
        task_short_description="adds two numbers",
        created_date="2025-07-09",
        created_by="test_user",
        main_fcn_code=ref_code,
        required_imports=[],
        fcn_dependency_code=[],
        reference_verification_inputs=[[2, 3], [0, 5]],
        test_cases=[
            {"test_code": test_code_1, "expected_failures": [fail_code_1, fail_code_2]},
            {"test_code": test_code_2, "expected_failures": [fail_code_1, fail_code_2]}
        ]
    )

    reference_fcn = compile_function_from_string(ref_code)
    results = evaluate_task_tests(task, reference_fcn)

    # --- Expected ---
    expected_reference = [
        ("test_add_positive", True),
        ("test_add_zero", True)
    ]
    expected_failures = [
        ("test_add_positive", True),
        ("test_add_positive", True),
        ("test_add_zero", True),
        ("test_add_zero", True)
    ]

    # --- Assertions ---
    assert sorted(results["reference_pass"]) == sorted(expected_reference)
    assert sorted(results["failure_fail"]) == sorted(expected_failures)

