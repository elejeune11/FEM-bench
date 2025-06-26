import builtins
import numpy as np
import pytest
from pathlib import Path
from pydantic import ValidationError
from fem_bench.yaml_load import Task, load_task, FunctionSignature, Parameter, CodeBlock, TestCase, ExpectedFailure  # Replace with your actual module
from fem_bench.yaml_load import load_all_tasks  # Include if needed
from fem_bench.yaml_load import get_type_string
from typing import Any
from tempfile import TemporaryDirectory
from typing import List, Dict, Tuple, Union
from typing import Callable, get_args

# ---------------------------------------------------------------------------
# UPDATE THIS IMPORT TO MATCH YOUR PACKAGE LAYOUT
# ---------------------------------------------------------------------------
from fem_bench.yaml_load import load_environment, Environment  # ← change path
# ---------------------------------------------------------------------------
from fem_bench.yaml_load import resolve_type, check_python_syntax
from fem_bench.yaml_load import extract_signature, get_source, create_task_from_functions, dump_task_to_yaml
import textwrap
import tempfile
import yaml
# ---------------------------------------------------------------------------
# 1 – A fully valid Tier-1 YAML ---------------------------------------------
# ---------------------------------------------------------------------------
VALID_YAML = """
environment_name: tier1_standard
tier: 1
description: Foundational FEM concepts
language: python
python_version: ">=3.8"

required_libraries:
  - name: numpy
    version: ">=1.20"
    import_as: np
    purpose: Numerical computations

  - name: pytest
    version: ">=6.0"
    purpose: Test framework

allowed_libraries:
  - name: math
    version: builtin
    purpose: Built-in maths

  - name: matplotlib
    version: ">=3.3"
    import_as: plt
    purpose: Plotting
    usage: when_needed

testing:
  framework: pytest
  required_imports:
    - "numpy as np"
    - pytest
  allowed_imports:
    - math
    - "matplotlib.pyplot as plt"
  naming_convention: test_*

code_requirements:
  max_function_length: 1000
  docstring_required: false
  vectorization_preferred: true

import_guidelines: |
  Use numpy and pytest by default.
"""


def test_load_valid_environment(tmp_path: Path):
    yaml_file = tmp_path / "env.yaml"
    yaml_file.write_text(VALID_YAML)

    env = load_environment(yaml_file)

    # top-level checks
    assert isinstance(env, Environment)
    assert env.environment_name == "tier1_standard"
    assert env.tier == 1

    # libraries
    assert any(lib.name == "numpy" for lib in env.required_libraries)
    assert any(lib.name == "matplotlib" and lib.import_as == "plt"
               for lib in env.allowed_libraries)

    # nested typed configs  (dot access, not dict indexing)
    assert env.testing is not None
    assert env.testing.framework == "pytest"

    assert env.code_requirements is not None
    assert env.code_requirements.max_function_length == 1000

# ---------------------------------------------------------------------------
# 2 – Missing a required top-level key  --------------------------------------
# ---------------------------------------------------------------------------
MISSING_KEY_YAML = """
tier: 1
description: No environment_name
language: python
python_version: ">=3.8"
required_libraries:
  - name: dummy
    version: ">=0"
    purpose: dummy
"""


def test_missing_required_key(tmp_path: Path):
    """Schema should reject YAML that omits environment_name."""
    f = tmp_path / "missing.yaml"
    f.write_text(MISSING_KEY_YAML)

    with pytest.raises(ValidationError):
        load_environment(f)


# ---------------------------------------------------------------------------
# 3 – Extra / unexpected key  ------------------------------------------------
# ---------------------------------------------------------------------------
EXTRA_KEY_YAML = VALID_YAML + "\nunexpected_field: oops\n"


def test_extra_key_disallowed(tmp_path: Path):
    """extra='forbid' should raise when an unknown field appears."""
    f = tmp_path / "extra.yaml"
    f.write_text(EXTRA_KEY_YAML)

    with pytest.raises(ValidationError):
        load_environment(f)


# ---------------------------------------------------------------------------
# 4 – Wrong scalar type (string instead of int) ------------------------------
# ---------------------------------------------------------------------------
BAD_TYPE_YAML = VALID_YAML.replace("max_function_length: 1000",
                                   'max_function_length: "many"')


def test_wrong_scalar_type(tmp_path: Path):
    """Int field receiving a string triggers ValidationError."""
    f = tmp_path / "badtype.yaml"
    f.write_text(BAD_TYPE_YAML)

    with pytest.raises(ValidationError):
        load_environment(f)


# ---------------------------------------------------------------------------
# 5 – File does not exist ----------------------------------------------------
# ---------------------------------------------------------------------------
def test_file_not_found():
    """Non-existent file path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_environment(Path("does_not_exist.yaml"))


TIER1_ENV_YAML = """\
# ============================================================================
# Tier 1 Standard Environment
# ============================================================================

# This file defines the standardized environment for all Tier 1 tasks
# Reference this in documentation and evaluation code

environment_name: "tier1_standard"
tier: 1
description: "Foundational FEM concepts using Python scientific stack"

# Language and version
language: "python"
python_version: ">=3.8"

# Required libraries (always available)
required_libraries:
  - name: "numpy"
    version: ">=1.20.0"
    import_as: "np"
    purpose: "Numerical computations and arrays"
  
  - name: "pytest" 
    version: ">=6.0.0"
    import_as: null
    purpose: "Test framework for LLM-generated tests"

# Allowed libraries (can be used when needed)
allowed_libraries:
  - name: "math"
    version: "builtin"
    import_as: null
    purpose: "Basic mathematical functions (sin, cos, sqrt, etc.)"
    
  - name: "scipy"
    version: ">=1.7.0"
    import_as: "scipy"
    purpose: "Advanced numerical methods (sparse matrices, optimization, etc.)"
    usage: "when_needed"    # Not required by default
    
  - name: "matplotlib"
    version: ">=3.3.0"
    import_as: "plt"
    purpose: "Plotting and visualization"
    usage: "when_needed"    # For tasks that require visualization

# Testing requirements
testing:
  framework: "pytest"
  required_imports: ["numpy as np", "pytest"]
  allowed_imports: ["math", "scipy", "matplotlib.pyplot as plt"]
  naming_convention: "test_*"
  
# Code requirements  
code_requirements:
  max_function_length: 1000      # Lines of code
  docstring_required: false
  vectorization_preferred: true # Use numpy operations over loops when possible
  
# Import guidelines
import_guidelines: |
  Standard imports for all tasks:
  - import numpy as np
  - import pytest
  
  Additional imports when needed:
  - import math
  - import scipy
  - import matplotlib.pyplot as plt
  
  LLMs should only import what they actually use in their implementation.
"""


def test_tier1_standard_env_parsing(tmp_path: Path):
    """Test parsing the full Tier 1 standard environment YAML."""
    yaml_file = tmp_path / "tier1.yaml"
    yaml_file.write_text(TIER1_ENV_YAML)

    env = load_environment(yaml_file)

    assert isinstance(env, Environment)
    assert env.environment_name == "tier1_standard"
    assert env.tier == 1
    assert env.language == "python"
    assert env.python_version == ">=3.8"
    assert len(env.required_libraries) == 2
    assert len(env.allowed_libraries) == 3

    numpy_lib = next(lib for lib in env.required_libraries if lib.name == "numpy")
    assert numpy_lib.import_as == "np"

    pytest_lib = next(lib for lib in env.required_libraries if lib.name == "pytest")
    assert pytest_lib.import_as is None

    assert env.testing is not None
    assert env.testing.framework == "pytest"
    assert "numpy as np" in env.testing.required_imports

    assert env.code_requirements.max_function_length == 1000
    assert env.code_requirements.docstring_required is False
    assert env.code_requirements.vectorization_preferred is True

    assert "import numpy as np" in env.import_guidelines


def test_resolve_type_valid_builtin_types():
    assert resolve_type("int") is int
    assert resolve_type("float") is float
    assert resolve_type("str") is str
    assert resolve_type("list") is list
    assert resolve_type("dict") is dict
    assert resolve_type("tuple") is tuple


def test_resolve_type_valid_external_type():
    assert resolve_type("numpy.ndarray") == np.ndarray


def test_resolve_type_invalid_type_string():
    with pytest.raises(ValueError, match="Unsupported type string"):
        resolve_type("undefined_type")

    with pytest.raises(ValueError, match="Unsupported type string"):
        resolve_type("int(")  # malformed expression

    with pytest.raises(ValueError, match="Unsupported type string"):
        resolve_type("os.system('rm -rf /')")  # security test


def test_resolve_type_callable_support():
    fn = resolve_type("lambda x: x + 1")
    assert callable(fn)
    assert fn(3) == 4

    # def_str = "def foo(): return 42"
    # Note: `eval()` won't work for full def statements. Those require `exec()`, not supported here.
    # So stick to lambdas for callables.


def test_valid_python_code_snippets():
    valid_snippets = [
        "def foo():\n    return 42",
        "for i in range(5):\n    print(i)",
        "x = [1, 2, 3]",
        "if True:\n    pass",
        "class A:\n    def method(self):\n        return None"
    ]
    for code in valid_snippets:
        check_python_syntax(code)  # Should not raise


def test_invalid_syntax_missing_colon():
    code = "def bad()\n    return 1"
    with pytest.raises(ValueError, match="Invalid Python syntax"):
        check_python_syntax(code, label="missing_colon_test")


def test_invalid_syntax_bad_indentation():
    code = "if True:\nprint('no indent')"
    with pytest.raises(ValueError, match="Invalid Python syntax"):
        check_python_syntax(code, label="indentation_test")


def test_invalid_syntax_malformed_expression():
    code = "x = [1, 2, 3"
    with pytest.raises(ValueError, match="Invalid Python syntax"):
        check_python_syntax(code, label="bracket_test")


def test_error_message_includes_label():
    code = "def broken("
    label = "unit_test_block"
    with pytest.raises(ValueError) as excinfo:
        check_python_syntax(code, label=label)
    assert label in str(excinfo.value)


@pytest.mark.parametrize(
    "type_str,expected_base,expected_dtype",
    [
        ("numpy.ndarray[float]", np.ndarray, float),
        ("numpy.ndarray[int]", np.ndarray, int),
        ("numpy.ndarray[str]", np.ndarray, str),
        ("numpy.ndarray[bool]", np.ndarray, bool),
        ("np.ndarray[float]", np.ndarray, float),
    ]
)
def test_resolve_type_numpy_array_with_entry_type(type_str, expected_base, expected_dtype):
    result = resolve_type(type_str)
    assert isinstance(result, tuple)
    assert result[0] is expected_base
    assert result[1] is expected_dtype


def test_resolve_type_numpy_array_without_entry_type():
    result = resolve_type("numpy.ndarray")
    assert result is np.ndarray


def test_resolve_type_invalid_numpy_array_inner_type():
    with pytest.raises(ValueError):
        resolve_type("numpy.ndarray[undefinedtype]")


def test_resolve_type_wrong_base_type_with_index():
    with pytest.raises(ValueError):
        resolve_type("list[int]")  # Only numpy.ndarray indexing is supported


def test_resolve_type_partial_brackets():
    with pytest.raises(ValueError):
        resolve_type("numpy.ndarray[")  # malformed


def test_resolve_type_nested_indexing_not_supported():
    with pytest.raises(ValueError):
        resolve_type("numpy.ndarray[numpy.ndarray[float]]")  # currently unsupported


def make_valid_task_dict():
    return {
        "task_id": "T1_SF_001",
        "category": "shape_functions",
        "subcategory": "linear",
        "title": "Linear Shape Functions",
        "short_description": "Compute linear shape functions for 1D element.",
        "version": "1.0",
        "created_date": "2025-06-23",
        "created_by": "test_user",
        "prompt_description": "Write a function to compute shape functions...",
        "expected_function_name": "shape_functions",
        "include_tests": True,
        "expected_test_functions": ["test_shape_functions"],
        "function_signature": {
            "input_parameters": [
                {"name": "xi", "type": "float"}
            ],
            "return_parameters": [
                {"name": "N", "type": "numpy.ndarray", "shape": "(2,)"}
            ]
        },
        "task_dependencies": {
            "required_functions": []
        },
        "reference_solution": {
            "code": "def shape_functions(xi):\n    return np.array([1 - xi, xi])"
        },
        "failure_examples": {
            "wrong_shape": {
                "code": "def shape_functions(xi):\n    return xi"
            }
        },
        "reference_verification": {
            "test_cases": [
                {
                    "input": {
                        "xi": 0.5
                    },
                    "tolerance": 1e-12
                }
            ]
        },
        "test_efficacy_verification": {
            "expected_failures": [
                {
                    "failure_type": "wrong_shape",
                    "test_function": "test_shape_functions"
                }
            ]
        }
    }


def test_task_model_validation_passes():
    task_data = make_valid_task_dict()
    task = Task.model_validate(task_data)
    assert task.task_id == "T1_SF_001"
    assert task.function_signature.input_parameters[0].name == "xi"


def test_task_model_validation_fails_on_wrong_test_input():
    task_data = make_valid_task_dict()
    task_data["reference_verification"]["test_cases"][0]["input"] = {"invalid_param": 1.0}

    with pytest.raises(ValueError, match="unexpected parameters"):
        Task.model_validate(task_data)


def test_task_model_validation_fails_on_bad_code():
    task_data = make_valid_task_dict()
    task_data["reference_solution"]["code"] = "def bad(:"

    with pytest.raises(ValueError, match="Invalid Python syntax"):
        Task.model_validate(task_data)


def test_load_valid_yaml_task():
    # Compute path relative to this test file
    fixture_path = Path(__file__).parent / "fixtures" / "T1_EX_001.yaml"
    assert fixture_path.exists(), "Fixture file not found"

    task = load_task(fixture_path)

    assert isinstance(task, Task)
    assert task.task_id == "T1_EX_001"
    assert task.expected_function_name == "add_two_numbers"
    assert len(task.reference_verification["test_cases"]) == 2


def fn_simple(x: float) -> int:
    return int(x)


def fn_multiple(x: float, y: int) -> float:
    return x + y


def fn_missing_type(x, y: float):  # no return type
    return x * y


def fn_any_type(x: Any) -> Any:
    return x


def fn_ndarray(x: float) -> np.ndarray:
    return np.array([x, x])


# ------------------------
# Test cases
# ------------------------

def test_extract_signature_simple():
    sig = extract_signature(fn_simple)
    assert isinstance(sig, FunctionSignature)
    assert len(sig.input_parameters) == 1
    assert sig.input_parameters[0].name == "x"
    assert sig.input_parameters[0].type == "float"
    assert sig.return_parameters[0].type == "int"


def test_extract_signature_multiple_params():
    sig = extract_signature(fn_multiple)
    assert [p.name for p in sig.input_parameters] == ["x", "y"]
    assert [p.type for p in sig.input_parameters] == ["float", "int"]
    assert sig.return_parameters[0].type == "float"


def test_extract_signature_missing_types_raises():
    with pytest.raises(ValueError, match="missing a type annotation"):
        extract_signature(fn_missing_type)

def test_extract_signature_with_any_raises():
    with pytest.raises(ValidationError, match="Unsupported type string: typing.Any"):
        extract_signature(fn_any_type)


def test_extract_signature_ndarray():
    sig = extract_signature(fn_ndarray)
    assert sig.input_parameters[0].type == "float"
    assert sig.return_parameters[0].type == "numpy.ndarray"  # Note: __name__ of np.ndarray is "ndarray"


def simple_function(x):
    return x + 1


def test_get_source_top_level():
    source = get_source(simple_function)
    expected = textwrap.dedent("""\
        def simple_function(x):
            return x + 1
    """)
    assert source.strip() == expected.strip()


# ---------------------------------------------------------
# Sample functions for test input
# ---------------------------------------------------------

def reference_fn(xi: float) -> np.ndarray:
    return np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])


def bad_formula(xi: float) -> np.ndarray:
    return xi  # invalid return type


def bad_shape(xi: float) -> np.ndarray:
    return np.array([[xi, xi]])


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------
def test_create_task_from_reference_and_failures():
    task = create_task_from_functions(
        task_id="T1_TEST_001",
        reference_fn=reference_fn,
        failure_fns={"wrong_formula": bad_formula, "wrong_shape": bad_shape},
        title="1D Shape Function",
        prompt_description="Compute 1D linear shape functions"
    )

    # Basic assertions
    assert isinstance(task, Task)
    assert task.task_id == "T1_TEST_001"
    assert task.expected_function_name == "reference_fn"
    assert task.function_signature.input_parameters[0].name == "xi"
    assert task.function_signature.input_parameters[0].type == "float"
    assert task.function_signature.return_parameters[0].type == "numpy.ndarray"

    # Failure examples exist
    assert "wrong_formula" in task.failure_examples
    assert "wrong_shape" in task.failure_examples
    assert "test_reference_fn" in task.expected_test_functions

    # Reference test cases populated
    assert len(task.reference_verification["test_cases"]) == 1
    assert task.reference_verification["test_cases"][0].input["xi"] == 0.0
    assert task.reference_verification["test_cases"][0].tolerance == 1e-12

    # Failure mapping is correct
    failure_types = [f.failure_type for f in task.test_efficacy_verification["expected_failures"]]
    assert set(failure_types) == {"wrong_formula", "wrong_shape"}


# ---------------------------------------------------------
# Override test
# ---------------------------------------------------------

def test_create_task_with_override():
    def ref(x: float) -> float:
        return x + 1

    def bad(x: float) -> float:
        return 0

    task = create_task_from_functions(
        task_id="T1_OVERRIDE",
        reference_fn=ref,
        failure_fns={"f": bad},
        short_description="original",
        reference_verification={
            "test_cases": [
                {"input": {"x": 3.0}, "tolerance": 1e-8}
            ]
        }
    )

    # short_description should be overridden
    assert task.short_description == "original"
    assert task.reference_verification["test_cases"][0].input["x"] == 3.0


def bad_fn(xi: float) -> float:
    return 0


def test_dump_task_to_yaml_creates_valid_file():
    # Step 1: Create a Task
    task = create_task_from_functions(
        task_id="T1_YAML_001",
        reference_fn=reference_fn,
        failure_fns={"bad": bad_fn},
        created_by="pytest"
    )

    # Step 2: Write to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "task.yaml"
        dump_task_to_yaml(task, path)

        # Step 3: Confirm file was created
        assert path.exists()

        # Step 4: Load it back as YAML
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        # Step 5: Validate structure
        assert isinstance(loaded, dict)
        assert loaded["task_id"] == "T1_YAML_001"
        assert loaded["expected_function_name"] == "reference_fn"
        assert "reference_solution" in loaded
        assert "code" in loaded["reference_solution"]

        # Optional: validate it reloads into a Task (roundtrip)
        rehydrated = Task.model_validate(loaded)
        assert rehydrated.task_id == task.task_id


def ref(xi: float) -> float:
    return xi + 1


def bad_wrong_args(x: float) -> float:  # wrong arg name
    return x


def test_failure_example_argument_mismatch_raises():
    with pytest.raises(ValidationError, match="has argument names .* expected .*"):
        create_task_from_functions(
            task_id="T1_FAIL_001",
            reference_fn=ref,
            failure_fns={"bad_args": bad_wrong_args}
        )


def bad_wrong_name(xi: float) -> float:
    return xi * 2


def test_failure_example_function_name_is_rewritten():
    task = create_task_from_functions(
        task_id="T1_REWRITE_FN",
        reference_fn=ref,
        failure_fns={"bad_name": bad_wrong_name}
    )

    # Confirm that the function name was changed to match expected_function_name
    code = task.failure_examples["bad_name"].code
    assert code.startswith("def ref(")


def test_test_case_input_validation():
    with pytest.raises(ValidationError, match="unexpected parameters"):
        create_task_from_functions(
            task_id="T1_BAD_INPUTS",
            reference_fn=ref,
            failure_fns={},
            reference_verification={
                "test_cases": [
                    {"input": {"wrong_param": 1.0}, "tolerance": 1e-12}
                ]
            }
        )


# ============================================================================
# Batch Loading Tests
# ============================================================================

def test_load_all_tasks_success(tmp_path):
    # Create dummy valid tasks
    task1 = tmp_path / "T1_A.yaml"
    task1.write_text("""
    task_id: "T1_A"
    category: "test"
    subcategory: "unit"
    title: "Task A"
    short_description: "Test A"
    version: "1.0"
    created_date: "2025-01-01"
    created_by: "test"
    prompt_description: "Do something"
    expected_function_name: "func_a"
    include_tests: false
    expected_test_functions: []
    function_signature:
      input_parameters: []
      return_parameters:
        - name: "out"
          type: "float"
          shape: ""
    task_dependencies:
      required_functions: []
    reference_solution:
      code: "def func_a(): return 0.0"
    failure_examples: {}
    reference_verification:
      test_cases: []
    test_efficacy_verification:
      expected_failures: []
    """)
    
    tasks = load_all_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].task_id == "T1_A"


def test_load_all_tasks_skips_templates_and_ignores_errors(tmp_path):
    # Template file (should be skipped)
    (tmp_path / "_TEMPLATE.yaml").write_text("")

    # Malformed file (should be caught and logged)
    (tmp_path / "bad.yaml").write_text("not: valid: yaml:")

    # Valid file
    (tmp_path / "T1_VALID.yaml").write_text("""
    task_id: "T1_VALID"
    category: "test"
    subcategory: "unit"
    title: "Valid Task"
    short_description: "Test task"
    version: "1.0"
    created_date: "2025-01-01"
    created_by: "tester"
    prompt_description: "Do the thing"
    expected_function_name: "valid_func"
    include_tests: false
    expected_test_functions: []
    function_signature:
      input_parameters: []
      return_parameters:
        - name: "out"
          type: "float"
          shape: ""
    task_dependencies:
      required_functions: []
    reference_solution:
      code: "def valid_func(): return 0.0"
    failure_examples: {}
    reference_verification:
      test_cases: []
    test_efficacy_verification:
      expected_failures: []
    """)

    tasks = load_all_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].task_id == "T1_VALID"


def test_builtin_types():
    assert get_type_string(int) == "int"
    assert get_type_string(float) == "float"
    assert get_type_string(bool) == "bool"


def test_numpy_type():
    assert get_type_string(np.ndarray) == "numpy.ndarray"


def test_list_type():
    assert get_type_string(List[int]) == "List[int]"
    assert get_type_string(List[Dict[str, float]]) == "List[Dict[str, float]]"


def test_dict_type():
    assert get_type_string(Dict[str, float]) == "Dict[str, float]"


def test_callable_type():
    t = Callable[[float], float]

    # `get_args` for Callable returns ( [arg_list], return_type )
    assert get_args(t) == ([float], float)

    # and your pretty-printer should still work
    assert get_type_string(t) == "Callable[[float], float]"


def test_fallback_str():
    class Custom:
        pass
    assert get_type_string(Custom) == "Custom"


def test_unsupported_or_nested_defaults():
    # This should fallback gracefully to string
    assert "Union" in get_type_string(Union[int, float])
    assert "Tuple" in get_type_string(Tuple[int, int])