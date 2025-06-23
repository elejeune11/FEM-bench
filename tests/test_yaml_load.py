"""
Tests for fem_bench.loader module.
"""

import pytest
import yaml
from pathlib import Path
from tempfile import TemporaryDirectory
from fem_bench.yaml_load import (
    load_task, load_environment, load_all_tasks,
    Task, Environment, Library, FunctionSignature, ReferenceSolution
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_task_yaml():
    """Sample task YAML content."""
    return """
task_id: "T1_SF_001"
category: "shape_functions"
subcategory: "linear"
title: "1D Linear Shape Functions"
short_description: "Compute linear shape functions for 1D two-node element"
version: "1.0"
created_date: "2025-06-04"
created_by: "elejeune11"

prompt: |
  Implement a function to compute 1D linear shape functions.

expected_function_name: "compute_1d_linear_shape_functions"
include_tests: true
expected_test_functions: ["test_partition_of_unity", "test_interpolation_property"]

function_signature:
  parameters: ["xi"]
  parameter_types: ["float"]
  return_shape: "(2,)"

task_dependencies:
  required_functions: []

reference_solution:
  code: |
    def compute_1d_linear_shape_functions(xi):
        N1 = (1.0 - xi) / 2.0
        N2 = (1.0 + xi) / 2.0
        return np.array([N1, N2])

failure_examples:
  "wrong_signs": |
    def compute_1d_linear_shape_functions(xi):
        N1 = (1.0 + xi) / 2.0
        N2 = (1.0 - xi) / 2.0
        return np.array([N1, N2])

reference_verification:
  test_cases:
    - input: {"xi": -1.0}
      tolerance: 1e-12
    - input: {"xi": 0.0}
      tolerance: 1e-12

test_efficacy_verification:
  expected_failures:
    - failure_type: "wrong_signs"
      test_function: "test_interpolation_property"
"""


@pytest.fixture
def sample_environment_yaml():
    """Sample environment YAML content."""
    return """
environment_name: "tier1_standard"
tier: 1
description: "Foundational FEM concepts using Python scientific stack"
language: "python"
python_version: ">=3.8"

required_libraries:
  - name: "numpy"
    version: ">=1.20.0"
    import_as: "np"
    purpose: "Numerical computations and arrays"
  
  - name: "pytest" 
    version: ">=6.0.0"
    import_as: null
    purpose: "Test framework for LLM-generated tests"

allowed_libraries:
  - name: "math"
    version: "builtin"
    import_as: null
    purpose: "Basic mathematical functions"
    usage: "when_needed"

testing:
  framework: "pytest"
  required_imports: ["numpy as np", "pytest"]
  naming_convention: "test_*"

code_requirements:
  max_function_length: 100
  docstring_required: true

import_guidelines: |
  Standard imports for all tasks:
  - import numpy as np
  - import pytest
"""


@pytest.fixture
def temp_task_file(sample_task_yaml):
    """Create a temporary task file."""
    with TemporaryDirectory() as tmpdir:
        task_file = Path(tmpdir) / "T1_SF_001.yaml"
        task_file.write_text(sample_task_yaml)
        yield task_file


@pytest.fixture
def temp_env_file(sample_environment_yaml):
    """Create a temporary environment file."""
    with TemporaryDirectory() as tmpdir:
        env_file = Path(tmpdir) / "tier1_environment.yaml"
        env_file.write_text(sample_environment_yaml)
        yield env_file


@pytest.fixture
def temp_tasks_dir(sample_task_yaml):
    """Create a temporary directory with multiple task files."""
    with TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir)
        
        # Create multiple task files
        (tasks_dir / "T1_SF_001.yaml").write_text(sample_task_yaml)
        
        # Create a second task (minimal version)
        task2_yaml = sample_task_yaml.replace("T1_SF_001", "T1_SF_002").replace("1D Linear", "2D Linear")
        (tasks_dir / "T1_SF_002.yaml").write_text(task2_yaml)
        
        # Create a template file (should be skipped)
        (tasks_dir / "_template.yaml").write_text(sample_task_yaml)
        
        yield tasks_dir


# ============================================================================
# Task Loading Tests
# ============================================================================

def test_load_task_success(temp_task_file):
    """Test successful task loading."""
    task = load_task(temp_task_file)
    assert isinstance(task, Task)
    assert task.task_id == "T1_SF_001"
    assert task.category == "shape_functions"
    assert task.subcategory == "linear"
    assert task.title == "1D Linear Shape Functions"
    assert task.expected_function_name == "compute_1d_linear_shape_functions"
    assert task.include_tests is True
    assert len(task.expected_test_functions) == 2


def test_load_task_function_signature(temp_task_file):
    """Test function signature parsing."""
    task = load_task(temp_task_file)
    sig = task.function_signature
    assert isinstance(sig, FunctionSignature)
    assert sig.parameters == ["xi"]
    assert sig.parameter_types == ["float"]
    assert sig.return_shape == "(2,)"


def test_load_task_reference_solution(temp_task_file):
    """Test reference solution parsing."""
    task = load_task(temp_task_file)
    
    # Test that reference_solution is a ReferenceSolution object
    assert isinstance(task.reference_solution, ReferenceSolution)
    
    # Test the code content
    assert "def compute_1d_linear_shape_functions" in task.reference_solution.code
    
    # Test failure examples
    assert "wrong_signs" in task.failure_examples
    assert "def compute_1d_linear_shape_functions" in task.failure_examples["wrong_signs"]


def test_load_task_test_cases(temp_task_file):
    """Test test cases parsing."""
    task = load_task(temp_task_file)
    
    assert len(task.test_cases) == 2
    assert task.test_cases[0].input == {"xi": -1.0}
    assert task.test_cases[0].tolerance == 1e-12


def test_load_task_expected_failures(temp_task_file):
    """Test expected failures parsing."""
    task = load_task(temp_task_file)
    
    assert len(task.expected_failures) == 1
    assert task.expected_failures[0].failure_type == "wrong_signs"
    assert task.expected_failures[0].test_function == "test_interpolation_property"


def test_load_task_file_not_found():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_task(Path("nonexistent.yaml"))


def test_load_task_missing_required_fields():
    """Test loading task with missing required fields."""
    with TemporaryDirectory() as tmpdir:
        bad_task_file = Path(tmpdir) / "bad_task.yaml"
        bad_task_file.write_text("task_id: 'T1_TEST'\n# missing other required fields")
        
        with pytest.raises(ValueError) as exc_info:
            load_task(bad_task_file)
        
        assert "Missing required fields" in str(exc_info.value)


def test_load_task_invalid_yaml():
    """Test loading invalid YAML."""
    with TemporaryDirectory() as tmpdir:
        bad_yaml_file = Path(tmpdir) / "bad.yaml"
        bad_yaml_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            load_task(bad_yaml_file)


# ============================================================================
# Environment Loading Tests
# ============================================================================

def test_load_environment_success(temp_env_file):
    """Test successful environment loading."""
    env = load_environment(temp_env_file)
    
    assert isinstance(env, Environment)
    assert env.environment_name == "tier1_standard"
    assert env.tier == 1
    assert env.language == "python"
    assert env.python_version == ">=3.8"


def test_load_environment_required_libraries(temp_env_file):
    """Test required libraries parsing."""
    env = load_environment(temp_env_file)
    
    assert len(env.required_libraries) == 2
    
    numpy_lib = next(lib for lib in env.required_libraries if lib.name == "numpy")
    assert isinstance(numpy_lib, Library)
    assert numpy_lib.version == ">=1.20.0"
    assert numpy_lib.import_as == "np"
    assert numpy_lib.purpose == "Numerical computations and arrays"
    
    pytest_lib = next(lib for lib in env.required_libraries if lib.name == "pytest")
    assert pytest_lib.import_as is None


def test_load_environment_allowed_libraries(temp_env_file):
    """Test allowed libraries parsing."""
    env = load_environment(temp_env_file)
    
    assert len(env.allowed_libraries) == 1
    
    math_lib = env.allowed_libraries[0]
    assert math_lib.name == "math"
    assert math_lib.version == "builtin"
    assert math_lib.usage == "when_needed"


def test_load_environment_testing_config(temp_env_file):
    """Test testing configuration parsing."""
    env = load_environment(temp_env_file)
    
    assert env.testing["framework"] == "pytest"
    assert "numpy as np" in env.testing["required_imports"]
    assert env.testing["naming_convention"] == "test_*"


def test_load_environment_code_requirements(temp_env_file):
    """Test code requirements parsing."""
    env = load_environment(temp_env_file)
    
    assert env.code_requirements["max_function_length"] == 100
    assert env.code_requirements["docstring_required"] is True


def test_load_environment_file_not_found():
    """Test loading non-existent environment file."""
    with pytest.raises(FileNotFoundError):
        load_environment(Path("nonexistent_env.yaml"))


def test_load_environment_missing_required_fields():
    """Test loading environment with missing required fields."""
    with TemporaryDirectory() as tmpdir:
        bad_env_file = Path(tmpdir) / "bad_env.yaml"
        bad_env_file.write_text("environment_name: 'test'\n# missing other required fields")
        
        with pytest.raises(ValueError) as exc_info:
            load_environment(bad_env_file)
        
        assert "Missing required fields" in str(exc_info.value)


# ============================================================================
# Batch Loading Tests
# ============================================================================

def test_load_all_tasks_success(temp_tasks_dir):
    """Test loading all tasks from directory."""
    tasks = load_all_tasks(temp_tasks_dir)
    
    assert len(tasks) == 2  # Should skip _template.yaml
    assert all(isinstance(task, Task) for task in tasks)
    assert tasks[0].task_id == "T1_SF_001"  # Should be sorted
    assert tasks[1].task_id == "T1_SF_002"


def test_load_all_tasks_empty_directory():
    """Test loading from empty directory."""
    with TemporaryDirectory() as tmpdir:
        tasks = load_all_tasks(Path(tmpdir))
        assert tasks == []


def test_load_all_tasks_skips_template_files(temp_tasks_dir):
    """Test that template files (starting with _) are skipped."""
    tasks = load_all_tasks(temp_tasks_dir)
    task_ids = [task.task_id for task in tasks]
    
    # Should not include any tasks from _template.yaml
    assert all("template" not in task_id.lower() for task_id in task_ids)


# ============================================================================
# Integration Tests
# ============================================================================

def test_task_and_environment_together(temp_task_file, temp_env_file):
    """Test loading both task and environment together."""
    task = load_task(temp_task_file)
    env = load_environment(temp_env_file)
    
    # Both should load successfully
    assert task.task_id == "T1_SF_001"
    assert env.environment_name == "tier1_standard"
    
    # Environment should provide what task needs
    numpy_available = any(lib.name == "numpy" for lib in env.required_libraries)
    pytest_available = any(lib.name == "pytest" for lib in env.required_libraries)
    
    assert numpy_available  # Task needs numpy
    assert pytest_available  # Task includes tests