import builtins
from collections import deque
from fem_bench.evaluate_output import evaluate_function_output_match
from fem_bench.evaluate_output import load_function_from_code
from fem_bench.evaluate_output import evaluate_task_tests, compile_function_from_string, run_test_case, load_test_functions
from fem_bench.evaluate_output import _values_match_impl, _values_match, _serialize_value
from fem_bench.task_base import Task
import math
import numpy as np
import pytest
import textwrap
from typing import Any, Set
from types import SimpleNamespace


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
    match, detailed_results = evaluate_function_output_match(ref_add_and_scale, gen_add_and_scale_correct, inputs)
    assert match
    assert len(detailed_results) == 2
    assert all(result["match"] for result in detailed_results)
    assert all(result["error"] is None for result in detailed_results)


def test_evaluate_function_output_match_mismatch():
    inputs = [[1, 2], [3, 4]]
    match, detailed_results = evaluate_function_output_match(ref_add_and_scale, gen_add_and_scale_wrong, inputs)
    assert not match
    assert len(detailed_results) == 2
    assert not any(result["match"] for result in detailed_results)
    assert all(result["error"] is None for result in detailed_results)


def test_evaluate_function_output_match_exception():
    inputs = [[1, 2]]
    match, detailed_results = evaluate_function_output_match(ref_add_and_scale, gen_broken, inputs)
    assert not match
    assert len(detailed_results) == 1
    assert not detailed_results[0]["match"]
    assert detailed_results[0]["error"] is not None


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


def extract_function_name(code: str) -> str:
    import ast
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    raise ValueError("No function definition found.")


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


def test_load_function_defined_as_non_callable_raises_value_error():
    # This defines a function syntactically, but overwrites it with an int
    code = '''
def not_a_function():
    return 42

not_a_function = 5
'''
    with pytest.raises(ValueError, match="Function 'not_a_function' not defined after execution."):
        load_function_from_code(code)


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


def call(a, b, **kwargs) -> bool:
    return _values_match_impl(
        a, b,
        atol=kwargs.pop('atol', 1e-8),
        rtol=kwargs.pop('rtol', 1e-5),
        strict_types=kwargs.pop('strict_types', False),
        visited=kwargs.pop('visited', set()),
        depth=kwargs.pop('depth', 0),
        max_depth=kwargs.pop('max_depth', 10),
        **kwargs
    )


def test_scalar_numbers():
    assert call(1.0, 1.0)
    assert call(1.0, 1.000000001)
    assert not call(1.0, 2.0)


def test_numpy_arrays():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0])
    c = np.array([[1, 2], [3, 4]])
    d = np.array([[1, 2], [3, 5]])
    assert call(a, b)
    assert not call(c, d)
    assert call(np.array(['a', 'b']), np.array(['a', 'b']))


def test_lists_and_tuples():
    assert call([1, 2], [1, 2])
    assert call((1, 2), (1, 2))
    assert not call([1, 2], (1, 2), strict_types=True)
    assert not call([1, 2], [2, 1])


def test_sets():
    assert call({1, 2}, {2, 1}, strict_types=True)
    assert not call({1, 2}, {1, 2, 3}, strict_types=True)
    assert not call({1, 2}, [1, 2], strict_types=True)


def test_dicts():
    d1 = {'a': 1, 'b': 2}
    d2 = {'a': 1, 'b': 2}
    d3 = {'a': 1, 'b': 3}
    assert call(d1, d2)
    assert not call(d1, d3)
    assert not call(d1, {'a': 1})


def test_mixed_array_and_list():
    assert call(np.array([1.0, 2.0]), [1.0, 2.0])
    assert call([1.0, 2.0], np.array([1.0, 2.0]))
    assert not call(np.array([1.0, 2.0]), [1.0, 2.1])


def test_nested_and_recursive_structures():
    x = []
    x.append(x)
    y = []
    y.append(y)
    assert call(x, y)

    a = [1, [2, [3]]]
    b = [1, [2, [3]]]
    c = [1, [2, [4]]]
    assert call(a, b)
    assert not call(a, c)


def test_max_depth_exceeded():
    a = 0
    b = 0
    for _ in range(10):
        a = [a]
        b = [b]

    with pytest.raises(RecursionError):
        _values_match_impl(
            a, b,
            atol=1e-8,
            rtol=1e-5,
            strict_types=False,
            visited=set(),
            depth=0,
            max_depth=5
        )



def test_strings_and_iterables():
    assert call("abc", "abc")
    assert not call("abc", "abcd")
    assert call(deque([1, 2]), deque([1, 2]))
    assert not call(deque([1, 2]), deque([2, 1]))


def test_type_mismatch():
    assert not call(1, "1", strict_types=True)
    assert call(1, 1.0, strict_types=False)


def test_fallback_equality():
    class A:
        def __eq__(self, other): return True
    a, b = A(), A()
    assert call(a, b)


def test_numpy_arrays_nans_infs_and_strings():
    # NaNs treated equal with equal_nan=True
    a = np.array([1.0, math.nan, np.inf])
    b = np.array([1.0, math.nan, np.inf])
    assert call(a, b)

    # shape-mismatch branch
    assert not call(a.reshape(3, 1), b)

    # non-numeric dtype triggers np.array_equal fallback
    str_a = np.array(["foo", "bar"], dtype=object)
    str_b = np.array(["foo", "bar"], dtype=object)
    assert call(str_a, str_b)
    assert not call(str_a, np.array(["foo", "baz"], dtype=object))


def test_sets_numeric_tolerance_and_frozensets():
    # numeric tolerance inside _compare_sets
    s1 = {1.0, 2.0000000001}
    s2 = {1.0, 2.0}
    assert call(s1, s2)          # within default rtol

    # frozenset vs frozenset (ok) vs set (strict type)
    fr1 = frozenset({1, 2})
    fr2 = frozenset({2, 1})
    assert call(fr1, fr2)
    assert not call(fr1, {1, 2}, strict_types=True)


def test_generic_iterables_vs_list():
    dq = deque([1, 2, 3])
    lst = [1, 2, 3]
    # strict_types=False => OK
    assert call(dq, lst)
    # strict_types=True => fails in generic-iterable block
    assert not call(dq, lst, strict_types=True)


def test_bytes_and_strings():
    assert call(b"abc", b"abc")
    assert not call(b"abc", "abc")          # different types
    assert call("μ", "μ")                   # non-ascii equality


def test_recursive_dict_cycle_detection():
    d1, d2 = {}, {}
    d1["self"] = d1
    d2["self"] = d2
    # Should terminate without RecursionError and be equal
    assert call(d1, d2)


def test_array_vs_list_mismatch_after_coercion():
    a = np.array([1.0, 2.0, 3.0])
    bad = [1.0, 2.0, 4.0]
    assert not call(a, bad)


def test_dict_strict_type_value_diff():
    da = {"x": 1, "y": [1, 2]}
    db = {"x": 1, "y": (1, 2)}
    # strict_types matters in nested list/tuple comparison
    assert call(da, db)                     # loose
    assert not call(da, db, strict_types=True)


def test_numeric_relative_tolerance():
    # 1e6 vs 1e6 + 2e1 within rtol=1e-5  (20 ≪ 1e6*1e-5=10)
    assert call(1e6, 1e6 + 10)
    # outside tolerance
    assert not call(1e6, 1e6 + 1e3)


def test_depth_guard_on_fallback():
    # Create two identical objects differing only at deepest level
    a = b = 0
    for _ in range(6):
        a = [a]
        b = [b]
    a.append(1)          # force inequality so recursion reaches fallback
    b.append(1)
    with pytest.raises(RecursionError):
        _values_match_impl(a, b, atol=1e-8, rtol=1e-5,
                           strict_types=False, visited=set(),
                           depth=0, max_depth=5)   # inclusive guard triggers


def test_fallback_equality_objects():
    class AlwaysEqual:
        def __eq__(self, other): return True

    class NeverEqual:
        def __eq__(self, other): return False

    # ⬆ positive branch: fallback equality returns True
    assert call(AlwaysEqual(), AlwaysEqual())
    assert call(AlwaysEqual(), NeverEqual())          # AlwaysEqual says “True”

    # ⬇ negative branch: both objects report inequality
    assert not call(NeverEqual(), NeverEqual())


def test_array_vs_tuple_strict_types():
    arr = np.array([1, 2, 3])
    tpl = (1, 2, 3)
    assert call(arr, tpl)                  # coercion allowed
    assert not call(arr, tpl, strict_types=True)


def test_large_sets_branch():
    big1 = set(range(30))
    big2 = set(range(30))
    big3 = set(range(29)) | {999}

    # equality via the (>20) fast-path
    assert _values_match(big1, big2)

    # inequality on the same fast-path
    assert not _values_match(big1, big3)


def test_small_sets_numeric_mismatch():
    s1 = {1.0, 2.0}
    s2 = {1.0, 2.01}          # outside default rtol/atol
    assert not _values_match(s1, s2)


def test_array_vs_tuple_strict_types_mismatch():
    arr = np.array([1, 2, 3])
    tpl = (1, 2, 3)
    assert _values_match(arr, tpl)                # loose
    assert not _values_match(arr, tpl, strict_types=True)


def test_numeric_eq_nan_inf():
    assert _values_match(math.nan, math.nan)      # equal_nan=True
    assert not _values_match(np.inf, -np.inf)     # opposite signs


class WeirdIterable:
    def __init__(self, data): self.data = data
    def __iter__(self): return iter(self.data)

def test_other_iterable_strict_type_block():
    w1 = WeirdIterable([1, 2])
    w2 = WeirdIterable([1, 2])
    assert _values_match(w1, w2)                       # loose
    assert not _values_match(w1, [1, 2], strict_types=True)


def test_depth_guard_at_fallback():
    # Build two identical deep trees ending with unequal object so we
    # reach the final equality fallback after depth 6.
    a = b = 0
    for _ in range(6):
        a = [a]
        b = [b]
    a.append(1)
    b.append(2)

    with pytest.raises(RecursionError):
        _values_match_impl(a, b,
                           atol=1e-8, rtol=1e-5,
                           strict_types=False, visited=set(),
                           depth=0, max_depth=5)   # inclusive guard


def test_serialize_value_all_types():
    assert _serialize_value(np.array([[1, 2]]))["type"] == "ndarray"
    assert isinstance(_serialize_value(np.int64(5)), int)
    assert isinstance(_serialize_value(np.float64(3.14)), float)

    # callable branch
    def foo(): pass
    assert _serialize_value(foo) == "<function>"

    # list / tuple / dict recursion
    complex_obj = {"k": (1, [2, 3])}
    result = _serialize_value(complex_obj)
    assert result == {"k": [1, [2, 3]]}


def test_evaluate_function_output_match():
    # reference and good generated function
    def ref(x, y): return np.array([x + y])
    def good(x, y): return np.array([x + y])
    all_ok, details = evaluate_function_output_match(ref, good, [[1, 2], [3, 4]])
    assert all_ok and all(d["match"] for d in details)

    # a generated function that is wrong on one case
    def bad(x, y): return np.array([x + y + 1])
    all_ok, details = evaluate_function_output_match(ref, bad, [[1, 2], [3, 4]])
    assert not all_ok
    assert any(not d["match"] for d in details)


def test_numeric_edge_cases():
    # NaN must equal NaN with equal_nan=True
    assert _values_match_impl(math.nan, math.nan,
                              atol=1e-8, rtol=1e-5,
                              strict_types=False, visited=set(),
                              depth=0, max_depth=2)
    # +Inf vs –Inf should be unequal
    assert not _values_match_impl(math.inf, -math.inf,
                                  atol=1e-8, rtol=1e-5,
                                  strict_types=False, visited=set(),
                                  depth=0, max_depth=2)

def test_large_sets_fast_path():
    big1 = set(range(50))
    big2 = set(range(50))
    big3 = set(range(49)) | {999}
    assert _values_match_impl(big1, big2,
                              atol=1e-8, rtol=1e-5,
                              strict_types=True, visited=set(),
                              depth=0, max_depth=3)
    assert not _values_match_impl(big1, big3,
                                  atol=1e-8, rtol=1e-5,
                                  strict_types=True, visited=set(),
                                  depth=0, max_depth=3)


def test_array_tuple_coercion():
    arr = np.array([1, 2, 3])
    tpl = (1, 2, 3)
    # Coercion succeeds when strict_types=False
    assert _values_match_impl(arr, tpl,
                              atol=1e-8, rtol=1e-5,
                              strict_types=False, visited=set(),
                              depth=0, max_depth=3)
    # Fails when strict_types=True
    assert not _values_match_impl(arr, tpl,
                                  atol=1e-8, rtol=1e-5,
                                  strict_types=True, visited=set(),
                                  depth=0, max_depth=3)


class Weird:
    def __init__(self, data): self._data = data
    def __iter__(self): return iter(self._data)

def test_other_iterable_paths():
    wa = Weird([1, 2])
    wb = Weird([1, 2])
    assert _values_match_impl(wa, wb,
                              atol=1e-8, rtol=1e-5,
                              strict_types=False, visited=set(),
                              depth=0, max_depth=3)
    # comparing Weird to plain list with strict_types=True must fail
    assert not _values_match_impl(wa, [1, 2],
                                  atol=1e-8, rtol=1e-5,
                                  strict_types=True, visited=set(),
                                  depth=0, max_depth=3)


def test_serialize_value_remaining_branches():
    # ndarray branch
    nd = np.array([[7, 8]])
    ser = _serialize_value(nd)
    assert ser["type"] == "ndarray" and ser["data"] == [[7, 8]]

    # numpy scalar branches
    assert _serialize_value(np.int64(5)) == 5
    assert _serialize_value(np.float64(2.5)) == 2.5

    # callable branch
    def foo(): pass
    assert _serialize_value(foo) == "<function>"

    # list / tuple recursion
    tpl = (1, (2, [3]))
    assert _serialize_value(tpl) == [1, [2, [3]]]

    # dict recursion
    d = {"x": np.array([1])}
    out = _serialize_value(d)
    assert out["x"]["data"] == [1]


def test_none_handling():
    assert call(None, None)
    assert not call(None, 0)


def test_len_mismatch_lists():
    assert not call([1, 2], [1])

x = float("1.0")        # built at run-time → new object
y = float("1.0")        # new object again
assert x == y and not (x is y)


def test_numeric_eq_exception(monkeypatch):
    def boom(*_a, **_kw):
        raise TypeError("boom")
    monkeypatch.setattr(np, "isclose", boom)

    # now the identity short-circuit is skipped,
    # we enter the numeric-scalar block → _numeric_eq → boom → except → False
    assert not call(x, y)


def test_ragged_ndarray_vs_sequence():
    ragged = [[1], [1, 2]]             # np.asarray(ragged) raises ValueError
    a = np.array([1, 2])
    assert not call(a, ragged)
    assert not call(ragged, a)


def test_other_iterable_paths():
    assert not call(deque([1]), deque([1, 2]), strict_types=False)
    class BadIter:
        def __iter__(self):            # list(BadIter()) raises TypeError
            raise TypeError
    assert not call(BadIter(), BadIter())


def test_fallback_recursion_guard():
    obj1, obj2 = object(), object()
    with pytest.raises(RecursionError):
        _values_match_impl(obj1, obj2,
                           atol=1e-8, rtol=1e-5,
                           strict_types=False, visited=set(),
                           depth=0, max_depth=1)   # hits lines 208-212


def test_load_test_functions_and_compile():
    code = "def inc(x):\n    return x + 1"
    task = SimpleNamespace(required_imports=[], test_cases=[{"test_code": code}])
    fcns = load_test_functions(task)
    assert fcns[0][0] == "inc" and callable(fcns[0][1])


def test_evaluate_task_tests_error_branches():
    good_test = "def test_ok(f):\n    assert f(1) == 1"
    bad_test  = "def broken("          # syntax error → cannot compile

    task = SimpleNamespace(
        required_imports=[],
        test_cases=[
            {"test_code": bad_test},                     # loading fails
            {"test_code": good_test,
             "expected_failures": [bad_test]},           # loading fails again
        ]
    )

    def identity(x): return x
    results = evaluate_task_tests(task, identity)

    # first test couldn’t load → reference_pass is False
    assert results["reference_pass"][0][1] is False

    # all failure_fail entries must be False because the bad code never ran
    assert all(flag is False for _name, flag in results["failure_fail"])


def test_list_tuple_strict_type_mismatch():
    assert not _values_match_impl(
        [1, 2],               # list
        (1, 2),               # tuple
        atol=1e-8, rtol=1e-5,
        strict_types=True, visited=set(),
        depth=0, max_depth=10
    )


def test_set_frozenset_strict_type_mismatch():
    assert not _values_match_impl(
        {1, 2},                       # set
        frozenset({1, 2}),            # frozenset
        atol=1e-8, rtol=1e-5,
        strict_types=True, visited=set(),
        depth=0, max_depth=10
    )


class CustomIter:
    def __init__(self, data): self.data = data
    def __iter__(self): return iter(self.data)


def test_other_iterable_strict_type_mismatch():
    assert not _values_match_impl(
        CustomIter([1, 2]),           # custom iterable
        [1, 2],                       # plain list
        atol=1e-8, rtol=1e-5,
        strict_types=True, visited=set(),
        depth=0, max_depth=10
    )

