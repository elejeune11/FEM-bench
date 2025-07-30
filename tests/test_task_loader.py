from fem_bench.task_loader import load_task_from_info
from fem_bench.task_base import Task


def import_task_module():
    import sys
    from pathlib import Path

    # Dynamically import linear_uniform_mesh_1D.py from tests/files/
    test_dir = Path(__file__).parent
    files_dir = test_dir / "tasks_dir"
    sys.path.insert(0, str(files_dir))
    import linear_uniform_mesh_1D
    return linear_uniform_mesh_1D


def test_load_task_from_info_returns_valid_task():
    task_module = import_task_module()
    task = load_task_from_info(task_module.task_info)

    # --- Metadata checks ---
    assert isinstance(task, Task)
    assert task.task_id == "linear_uniform_mesh_1D"
    assert task.created_by == "elejeune11"
    assert task.created_date == "2025-07-08"
    assert "1D uniform mesh" in task.task_short_description

    # --- Main function code ---
    assert task.main_fcn_code.startswith("def linear_uniform_mesh_1D")
    assert "np.linspace" in task.main_fcn_code
    assert "element_connectivity" in task.main_fcn_code

    # --- Required imports and dependencies ---
    assert isinstance(task.required_imports, list)
    assert "import numpy as np" in task.required_imports
    assert isinstance(task.fcn_dependency_code, list)
    assert all(isinstance(dep, str) for dep in task.fcn_dependency_code)

    # --- Reference inputs ---
    assert isinstance(task.reference_verification_inputs, list)
    assert len(task.reference_verification_inputs) >= 1
    for input_set in task.reference_verification_inputs:
        assert isinstance(input_set, list)
        assert len(input_set) == 3  # expecting [x_min, x_max, num_elements]

    # --- Test cases ---
    assert isinstance(task.test_cases, list)
    for case in task.test_cases:
        assert "test_code" in case
        assert isinstance(case["test_code"], str)
        assert case["test_code"].startswith("def test_")
        assert "expected_failures" in case
        assert isinstance(case["expected_failures"], list)
        for fail_code in case["expected_failures"]:
            assert isinstance(fail_code, str)
            assert fail_code.startswith("def fail_")

    # --- Specific test case expectations (optional) ---
    test_names = [case["test_code"].split("(")[0] for case in task.test_cases]
    assert any("test_basic_mesh_creation" in name for name in test_names)
    assert any("test_single_element_mesh" in name for name in test_names)
