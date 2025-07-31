from fem_bench.task_base import CodeBlock, Task
import math
import pytest
import textwrap


def test_basic_signature_and_docstring():
    code = '''
def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b
'''
    cb = CodeBlock(main_function=code)
    sig = cb.get_signature()
    doc = cb.get_docstring()
    env = cb.execute()

    assert sig["name"] == "add"
    assert sig["args"] == [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}]
    assert sig["returns"] == "int"
    assert doc == "Adds two integers."
    assert env["add"](2, 3) == 5


def test_function_with_defaults():
    code = '''
def greet(name: str = "World") -> str:
    """Returns a greeting message."""
    return f"Hello, {name}!"
'''
    cb = CodeBlock(main_function=code)
    env = cb.execute()
    assert env["greet"]() == "Hello, World!"
    assert env["greet"]("Alice") == "Hello, Alice!"


def test_function_without_docstring():
    code = '''
def square(x: int) -> int:
    return x * x
'''
    cb = CodeBlock(main_function=code)
    assert cb.get_docstring() == ""


def test_unannotated_function():
    code = '''
def identity(x):
    """Returns the input as-is."""
    return x
'''
    cb = CodeBlock(main_function=code)
    sig = cb.get_signature()
    assert sig["args"] == [{"name": "x", "type": None}]
    assert sig["returns"] is None
    assert cb.get_docstring() == "Returns the input as-is."


def test_preamble_import_usage():
    preamble = 'import math'
    main = '''
def area(r: float) -> float:
    """Returns area of a circle."""
    return math.pi * r * r
'''
    cb = CodeBlock(main_function=main, preamble=preamble)
    env = cb.execute()
    result = env["area"](2.0)  # e.g., r = 2 → π * 4 ≈ 12.5664
    assert math.isclose(result, 12.56637, rel_tol=1e-6)


def test_missing_function_raises():
    bad_code = 'x = 42'
    with pytest.raises(ValueError, match="No function found"):
        CodeBlock(main_function=bad_code)


def test_malformed_main_code_raises():
    bad_code = 'def broken(:)'
    with pytest.raises(SyntaxError):
        CodeBlock(main_function=bad_code)


def test_malformed_preamble_raises_on_execute():
    preamble = 'def helper(x): return math.'
    main = 'def calc(x): return helper(x)'
    cb = CodeBlock(main_function=main, preamble=preamble)
    with pytest.raises(Exception):
        cb.execute()


def test_task_initialization_minimal():
    # Minimal function code (main and helper)
    main_fcn_code = textwrap.dedent("""
    def square_sum(a: int, b: int) -> int:
        \"\"\"Returns a² + b²\"\"\"
        return helper(a) + helper(b)
    """)

    helper_code = textwrap.dedent("""
    def helper(x: int) -> int:
        return x * x
    """)

    # Minimal test case with an expected failure
    test_cases = [
        {
            "test_code": textwrap.dedent("""
            def test_addition_incorrect(f):
                assert f(2, 3) == 13
            """),
            "expected_failures": [
                textwrap.dedent("""
                def bad_add(a, b):
                    return a + b
                """)
            ]
        }
    ]

    # Create task
    task = Task(
        task_id="t001",
        task_short_description="Sum of squares function",
        created_date="2025-07-09",
        created_by="benjamin",
        main_fcn_code=main_fcn_code,
        required_imports=["import math"],
        fcn_dependency_code=[helper_code],
        reference_verification_inputs=[[2, 3]],
        test_cases=test_cases
    )

    # Assertions
    assert task.task_id == "t001"
    assert task.created_by == "benjamin"
    assert "square_sum" in task.main_fcn_code
    assert "helper" in task.fcn_dependency_code[0]
    assert task.test_cases[0]["expected_failures"]
    assert isinstance(task.reference_verification_inputs, list)
    assert task.reference_verification_inputs[0] == [2, 3]


def test_task_repr():
    task = Task(
        task_id="001",
        task_short_description="Add two numbers",
        created_date="2025-07-30",
        created_by="Alice",
        main_fcn_code="def add(a, b): return a + b"
    )
    assert repr(task) == "<Task 001: Add two numbers>"