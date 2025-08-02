import ast
from fem_bench.task_base import Task
import numpy as np
import numbers
import pytest
from typing import Any, Callable, Optional, Set


def extract_function_name(code: str) -> str:
    """Extract the first function name defined in the given code block."""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    raise ValueError("No function definition found in the code.")


def load_function_from_code(
    code: str,
    required_imports: Optional[list[str]] = None,
    fcn_dependencies: Optional[list[str]] = None,
) -> Callable:
    """
    Load a Python function from a string with optional imports and dependencies.

    Returns the main function defined in `code`.
    """
    namespace = {}

    # Build full source
    parts = []
    if required_imports:
        parts.extend(required_imports)
    if fcn_dependencies:
        parts.extend(fcn_dependencies)
    parts.append(code)

    full_source = "\n\n".join(parts)
    exec(full_source, namespace)

    # Find the expected function name
    expected_name = extract_function_name(code)

    if expected_name not in namespace or not callable(namespace[expected_name]):
        raise ValueError(f"Function '{expected_name}' not defined after execution.")
    
    return namespace[expected_name]


def _values_match(
    a: Any,
    b: Any,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    strict_types: bool = False,
    max_depth: int = 100
) -> bool:
    """
    Recursively compare two values that can be scalars, NumPy arrays,
    sequences, or nested combinations thereof.

    Parameters
    ----------
    a, b : Any
        Values to compare
    atol : float, default 1e-8
        Absolute tolerance for numeric comparisons
    rtol : float, default 1e-5
        Relative tolerance for numeric comparisons
    strict_types : bool, default False
        If True, require exact type matches for containers (e.g., list != tuple)
    max_depth : int, default 100
        Maximum recursion depth to prevent infinite loops

    Returns
    -------
    bool
        True if *a* and *b* are equal within the given tolerances.
    
    Raises
    ------
    RecursionError
        If maximum recursion depth is exceeded
    """
    return _values_match_impl(a, b, atol=atol, rtol=rtol, strict_types=strict_types, 
                             visited=set(), depth=0, max_depth=max_depth)


def _values_match_impl(
    a: Any,
    b: Any,
    *,
    atol: float,
    rtol: float,
    strict_types: bool,
    visited: Set[tuple],
    depth: int,
    max_depth: int
) -> bool:
    """Compare two potentially-nested structures, with cycle detection and
    an inclusive recursion-depth limit (`depth >= max_depth` raises)."""

    # Inclusive depth guard
    if depth >= max_depth:
        raise RecursionError(f"Maximum recursion depth ({max_depth}) exceeded")

    # Short-circuit identity check
    if a is b:
        return True

    # None handling
    if a is None or b is None:
        return a is b

    # Cycle detection for every object pair
    identity_key = (id(a), id(b))
    if identity_key in visited:
        return True
    visited.add(identity_key)

    # Strict-type check
    if strict_types and type(a) is not type(b):
        return False

    # Helper for numeric scalars
    def _numeric_eq(x, y) -> bool:
        try:
            return np.isclose(x, y, atol=atol, rtol=rtol, equal_nan=True)
        except Exception:
            return False

    # --- NumPy arrays ----------------------------------------------------
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        try:
            return np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        except Exception:
            return np.array_equal(a, b)

    # --- Lists / tuples --------------------------------------------------
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(
            _values_match_impl(x, y, atol=atol, rtol=rtol, strict_types=strict_types,
                               visited=visited, depth=depth + 1, max_depth=max_depth)
            for x, y in zip(a, b)
        )

    # --- Sets / frozensets ----------------------------------------------
    if isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset)):
        if len(a) != len(b):
            return False
        return _compare_sets(a, b, atol=atol, rtol=rtol, strict_types=strict_types,
                             visited=visited, depth=depth + 1, max_depth=max_depth)

    # --- ndarray vs sequence --------------------------------------------
    if isinstance(a, np.ndarray) and isinstance(b, (list, tuple)):
        try:
            return _values_match_impl(a, np.asarray(b), atol=atol, rtol=rtol,
                                      strict_types=strict_types, visited=visited,
                                      depth=depth + 1, max_depth=max_depth)
        except Exception:
            return False
    if isinstance(b, np.ndarray) and isinstance(a, (list, tuple)):
        try:
            return _values_match_impl(np.asarray(a), b, atol=atol, rtol=rtol,
                                      strict_types=strict_types, visited=visited,
                                      depth=depth + 1, max_depth=max_depth)
        except Exception:
            return False

    # --- Numbers ---------------------------------------------------------
    if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
        return _numeric_eq(a, b)

    # --- Dicts -----------------------------------------------------------
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(
            _values_match_impl(a[k], b[k], atol=atol, rtol=rtol, strict_types=strict_types,
                               visited=visited, depth=depth + 1, max_depth=max_depth)
            for k in a
        )

    # --- Strings ---------------------------------------------------------
    if isinstance(a, str) and isinstance(b, str):
        return a == b

    # --- Other iterables (non-string) ------------------------------------
    if (hasattr(a, '__iter__') and hasattr(b, '__iter__') and
            not isinstance(a, (str, bytes)) and not isinstance(b, (str, bytes))):
        try:
            return _values_match_impl(list(a), list(b), atol=atol, rtol=rtol,
                                      strict_types=strict_types, visited=visited,
                                      depth=depth + 1, max_depth=max_depth)
        except Exception:
            return False

    # --- Fallback --------------------------------------------------------
    # One last inclusive guard in case we reach here after nesting deeper
    if depth + 1 >= max_depth:
        raise RecursionError(f"Maximum recursion depth ({max_depth}) exceeded at fallback")
    return a == b


def _compare_sets(
    a: set, 
    b: set, 
    *, 
    atol: float,
    rtol: float,
    strict_types: bool,
    visited: Set[tuple],
    depth: int,
    max_depth: int
) -> bool:
    """Compare sets with numeric tolerance."""
    # For small sets, do pairwise comparison
    if len(a) <= 20:  # Arbitrary threshold
        a_list = list(a)
        b_list = list(b)
        
        # Try to find a matching for each element in a
        used_b = set()
        for item_a in a_list:
            found_match = False
            for i, item_b in enumerate(b_list):
                if i in used_b:
                    continue
                if _values_match_impl(item_a, item_b, atol=atol, rtol=rtol,
                                    strict_types=strict_types, visited=visited,
                                    depth=depth+1, max_depth=max_depth):
                    used_b.add(i)
                    found_match = True
                    break
            if not found_match:
                return False
        return len(used_b) == len(b_list)
    else:
        # For large sets, fall back to exact equality
        return a == b


def evaluate_function_output_match(
    reference_fcn,
    generated_fcn,
    inputs: list[list],
    atol: float = 1e-8
) -> tuple[bool, list]:
    """
    Returns (match_result, detailed_results).
    
    match_result: True if all outputs from generated_fcn match those from reference_fcn
    detailed_results: List of dicts with inputs, reference output, generated output for each test
    """
    detailed_results = []
    all_match = True
    
    for i, input_args in enumerate(inputs):
        test_result = {
            "test_case": i + 1,
            "inputs": _serialize_inputs(input_args),
            "reference_output": None,
            "generated_output": None,
            "match": False,
            "error": None
        }
        
        try:
            ref_out = reference_fcn(*input_args)
            gen_out = generated_fcn(*input_args)
            
            test_result["reference_output"] = _serialize_output(ref_out)
            test_result["generated_output"] = _serialize_output(gen_out)
            
            match = _values_match(ref_out, gen_out, atol=atol)
            test_result["match"] = match
            
            if not match:
                all_match = False
                
        except Exception as e:
            test_result["error"] = str(e)
            all_match = False
        
        detailed_results.append(test_result)
    
    return all_match, detailed_results


def _serialize_inputs(input_args):
    """Serialize input arguments for JSON storage."""
    serialized = []
    for arg in input_args:
        serialized.append(_serialize_value(arg))
    return serialized


def _serialize_output(output):
    """Serialize output for JSON storage."""
    if isinstance(output, dict):
        return {k: _serialize_value(v) for k, v in output.items()}
    else:
        return _serialize_value(output)


def _serialize_value(value):
    """Serialize a single value for JSON storage."""
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "data": value.tolist()
        }
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif callable(value):
        return "<function>"
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    else:
        return value


def compile_function_from_string(code: str) -> Callable:
    local_ns = {}
    exec(code, {}, local_ns)
    return list(local_ns.values())[0]  # assume one function per string


def load_test_functions(task: Task) -> list[tuple[str, Callable]]:
    test_fcns = []
    for case in task.test_cases:
        f = compile_function_from_string(case["test_code"])
        test_fcns.append((f.__name__, f))
    return test_fcns


def run_test_case(test_fcn: Callable, fcn_under_test: Callable) -> bool:
    try:
        test_fcn(fcn_under_test)
        return True
    except (AssertionError, pytest.fail.Exception, Exception):
        return False


def evaluate_task_tests(task: Task, reference_fcn: Callable) -> dict:
    results = {"reference_pass": [], "failure_fail": []}

    for case in task.test_cases:
        try:
            test_fcn = load_function_from_code(
                case["test_code"],
                required_imports=task.required_imports
            )
        except Exception as e:
            results["reference_pass"].append((case.get("name", "unnamed_test"), False))
            continue

        # Check reference function passes
        passed = run_test_case(test_fcn, reference_fcn)
        results["reference_pass"].append((test_fcn.__name__, passed))

        # Check each expected failure fails
        for fail_code in case.get("expected_failures", []):
            try:
                failing_fcn = load_function_from_code(
                    fail_code,
                    required_imports=task.required_imports
                )
                failed = not run_test_case(test_fcn, failing_fcn)
            except Exception:
                failed = False
            results["failure_fail"].append((test_fcn.__name__, failed))

    return results

