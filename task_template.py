import numpy as np
from typing import List, Dict, Callable, Optional

# === Dependency functions (if any) ===
def helper_1(...): ...
def helper_2(...): ...

# === Reference implementation ===
def main_fcn(...):
    """Compute or solve something"""
    ...

# === Test functions ===
def test_case_1(fcn):
    """Docstring explaining what's tested."""
    ...

# === Known failing examples (optional) ===
def fail_case_1(...): ...
def fail_case_2(...): ...

# === task_info() metadata ===
def task_info():
    task_id = "unique_task_name"
    task_short_description = "concise description of what the task does"
    created_date = "YYYY-MM-DD"
    created_by = "your_name"

    main_fcn = main_fcn
    required_imports = [
        "import numpy as np",
        "import pytest",
        # any typing or math-related imports
    ]
    fcn_dependencies = [helper_1, helper_2]  # or [] if none

    reference_verification_inputs = [
        # List of lists: each sublist is a set of args for main_fcn
        [arg1, arg2, ...],
        ...
    ]

    test_cases = [
        {
            "test_code": test_case_1,
            "expected_failures": [fail_case_1, fail_case_2]  # or []
        },
        ...
    ]

    return (
        task_id,
        task_short_description,
        created_date,
        created_by,
        main_fcn,
        required_imports,
        fcn_dependencies,
        reference_verification_inputs,
        test_cases
    )
