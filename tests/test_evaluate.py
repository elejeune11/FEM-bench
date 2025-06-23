import json
import numpy as np
import pytest
from pathlib import Path
from fem_bench.evaluate import (
    build_full_dependency_tree,
    DependencyError,
    CircularDependencyError,
    extract_unique_nodes,
    DependencyNode,
    get_dependencies,
    build_environment_imports,
    build_dependency_code_reference,
    build_main_code_reference,
    build_dependency_code_llm,
    build_main_code_llm,
    aggregate_reference_code,
    aggregate_llm_code,
    evaluate_code_block,
    DependencyMode,
    _compare_outputs,
    VerificationResult,
    verify_against_reference,
    evaluate_task_implementation,
    TaskEvaluationResult,
    TestEvaluator,
    TestResult,
    TestQualityEvaluator,
    ReferenceTestResult,
    FailureDetectionResult,
    evaluate_single_task
)
from fem_bench.yaml_load import load_task, load_environment
from fem_bench.prompt import validate_llm_output, parse_llm_json_output


@pytest.fixture
def test_tasks_dir():
    """Get the test files directory for example tasks."""
    return Path(__file__).parent / "example_tasks"


def test_build_full_tree_no_deps(test_tasks_dir):
    """Test building full tree for task with no dependencies."""    
    task = load_task(test_tasks_dir / "T1_NO_DEPS.yaml")
    root_node = build_full_dependency_tree(task, test_tasks_dir)
    
    assert root_node.task_id == "T1_NO_DEPS"
    assert root_node.function_name == "standalone_function"
    assert len(root_node.dependencies) == 0
    assert root_node.task is not None


def test_build_full_tree_single_dependency(test_tasks_dir):
    """Test building full tree with single dependency."""    
    task = load_task(test_tasks_dir / "T1_CHAIN_B.yaml")  # Depends on T1_LEAF_A
    root_node = build_full_dependency_tree(task, test_tasks_dir)
    
    assert root_node.task_id == "T1_CHAIN_B"
    assert root_node.function_name == "intermediate_function"
    assert len(root_node.dependencies) == 1
    
    # Check the dependency
    dep_node = root_node.dependencies[0]
    assert dep_node.task_id == "T1_LEAF_A"
    assert dep_node.function_name == "function_a"
    assert len(dep_node.dependencies) == 0


def test_build_full_tree_chain(test_tasks_dir):
    """Test building full tree for chain dependency: A → B → C."""    
    task = load_task(test_tasks_dir / "T1_CHAIN.yaml")  # Fixed filename
    root_node = build_full_dependency_tree(task, test_tasks_dir)
    
    # T1_CHAIN → T1_CHAIN_B → T1_LEAF_A
    assert root_node.task_id == "T1_CHAIN"
    assert root_node.function_name == "chain_function"
    assert len(root_node.dependencies) == 1
    
    # Check first level dependency
    chain_b_node = root_node.dependencies[0]
    assert chain_b_node.task_id == "T1_CHAIN_B"
    assert chain_b_node.function_name == "intermediate_function"
    assert len(chain_b_node.dependencies) == 1
    
    # Check second level dependency
    leaf_a_node = chain_b_node.dependencies[0]
    assert leaf_a_node.task_id == "T1_LEAF_A"
    assert leaf_a_node.function_name == "function_a"
    assert len(leaf_a_node.dependencies) == 0
    
    # Verify all nodes have their task objects loaded
    assert root_node.task is not None
    assert chain_b_node.task is not None
    assert leaf_a_node.task is not None


def test_fanout_pattern_full_tree(test_tasks_dir):
    """Test fan-out pattern in full tree (should have no duplicates)."""    
    task = load_task(test_tasks_dir / "T1_FANOUT.yaml")
    root_node = build_full_dependency_tree(task, test_tasks_dir)
    
    # Should have 3 dependencies
    assert len(root_node.dependencies) == 3
    
    # Check that all dependencies are leaf nodes
    dep_task_ids = {dep.task_id for dep in root_node.dependencies}
    assert dep_task_ids == {"T1_LEAF_A", "T1_LEAF_B", "T1_LEAF_C"}
    
    # All should have no further dependencies
    for dep in root_node.dependencies:
        assert len(dep.dependencies) == 0


def test_build_full_tree_diamond_with_duplicates(test_tasks_dir):
    """Test that diamond pattern creates duplicate nodes in full tree."""    
    task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    root_node = build_full_dependency_tree(task, test_tasks_dir)
    
    # Root should have 2 dependencies (T1_DIAMOND_B and T1_DIAMOND_C)
    assert root_node.task_id == "T1_DIAMOND"
    assert len(root_node.dependencies) == 2
    
    # Find the two intermediate nodes
    diamond_b_node = None
    diamond_c_node = None
    for dep in root_node.dependencies:
        if dep.task_id == "T1_DIAMOND_B":
            diamond_b_node = dep
        elif dep.task_id == "T1_DIAMOND_C":
            diamond_c_node = dep
    
    assert diamond_b_node is not None
    assert diamond_c_node is not None
    
    # Both should depend on T1_LEAF_D
    assert len(diamond_b_node.dependencies) == 1
    assert diamond_b_node.dependencies[0].task_id == "T1_LEAF_D"
    
    assert len(diamond_c_node.dependencies) == 1
    assert diamond_c_node.dependencies[0].task_id == "T1_LEAF_D"
    
    # The T1_LEAF_D nodes should be separate objects (duplicates in full tree)
    leaf_d_from_b = diamond_b_node.dependencies[0]
    leaf_d_from_c = diamond_c_node.dependencies[0]
    
    # Same task_id but different object instances
    assert leaf_d_from_b.task_id == leaf_d_from_c.task_id == "T1_LEAF_D"
    assert leaf_d_from_b is not leaf_d_from_c  # Different objects


def test_circular_dependency_two_nodes(test_tasks_dir):
    """Test circular dependency detection with A → B → A."""    
    task = load_task(test_tasks_dir / "T1_CIRCULAR_A.yaml")
    
    with pytest.raises(CircularDependencyError) as exc_info:
        build_full_dependency_tree(task, test_tasks_dir)
    
    # Check that the error message contains the circular dependency path
    error_msg = str(exc_info.value)
    assert "Circular dependency detected" in error_msg
    assert "T1_CIRCULAR_A" in error_msg
    assert "T1_CIRCULAR_B" in error_msg


def test_missing_dependency_file(test_tasks_dir):
    """Test error when dependency file doesn't exist."""    
    task = load_task(test_tasks_dir / "T1_CHAIN_B.yaml")  # This has dependencies
    
    # Manually modify the task to depend on non-existent file
    task.task_dependencies[0].source_task = "NONEXISTENT_TASK"
    
    with pytest.raises(DependencyError):
        build_full_dependency_tree(task, test_tasks_dir)


def test_resolution_stack_cleanup(test_tasks_dir):
    """Test that resolution stack is properly cleaned up even after errors."""    
    task = load_task(test_tasks_dir / "T1_CIRCULAR_A.yaml")
    
    # First call should raise CircularDependencyError
    with pytest.raises(CircularDependencyError):
        build_full_dependency_tree(task, test_tasks_dir)
    
    # Second call should also raise CircularDependencyError (not some other error)
    # This verifies that the resolution_stack was properly cleaned up
    with pytest.raises(CircularDependencyError):
        build_full_dependency_tree(task, test_tasks_dir)


def test_extract_unique_nodes_no_deps():
    """Test extracting unique nodes from single node with no dependencies."""
    # Create a simple node with no dependencies
    root_node = DependencyNode(
        task_id="T1_SIMPLE",
        function_name="simple_func",
        dependencies=[]
    )
    
    unique_nodes = extract_unique_nodes(root_node)
    
    assert len(unique_nodes) == 1
    assert "T1_SIMPLE" in unique_nodes
    assert unique_nodes["T1_SIMPLE"].function_name == "simple_func"


def test_extract_unique_nodes_linear_chain():
    """Test extracting unique nodes from linear chain (no duplicates)."""
    # Create chain: A → B → C
    node_c = DependencyNode(task_id="C", function_name="func_c", dependencies=[])
    node_b = DependencyNode(task_id="B", function_name="func_b", dependencies=[node_c])
    node_a = DependencyNode(task_id="A", function_name="func_a", dependencies=[node_b])
    
    unique_nodes = extract_unique_nodes(node_a)
    
    assert len(unique_nodes) == 3
    assert set(unique_nodes.keys()) == {"A", "B", "C"}
    assert unique_nodes["A"].function_name == "func_a"
    assert unique_nodes["B"].function_name == "func_b"
    assert unique_nodes["C"].function_name == "func_c"


def test_extract_unique_nodes_diamond_pattern():
    """Test extracting unique nodes from diamond pattern (with duplicates)."""
    # Create diamond: A → B, C; B → D; C → D (D appears twice)
    node_d1 = DependencyNode(task_id="D", function_name="func_d", dependencies=[])
    node_d2 = DependencyNode(task_id="D", function_name="func_d", dependencies=[])  # Duplicate
    
    node_b = DependencyNode(task_id="B", function_name="func_b", dependencies=[node_d1])
    node_c = DependencyNode(task_id="C", function_name="func_c", dependencies=[node_d2])
    node_a = DependencyNode(task_id="A", function_name="func_a", dependencies=[node_b, node_c])
    
    unique_nodes = extract_unique_nodes(node_a)
    
    # Should have 4 unique nodes despite D appearing twice
    assert len(unique_nodes) == 4
    assert set(unique_nodes.keys()) == {"A", "B", "C", "D"}
    
    # Should be the first occurrence of D (from node_b path)
    assert unique_nodes["D"] is node_d1
    assert unique_nodes["D"] is not node_d2


def test_extract_unique_nodes_fan_out():
    """Test extracting unique nodes from fan-out pattern."""
    # Create fan-out: A → B, C, D (no duplicates)
    node_b = DependencyNode(task_id="B", function_name="func_b", dependencies=[])
    node_c = DependencyNode(task_id="C", function_name="func_c", dependencies=[])
    node_d = DependencyNode(task_id="D", function_name="func_d", dependencies=[])
    node_a = DependencyNode(task_id="A", function_name="func_a", dependencies=[node_b, node_c, node_d])
    
    unique_nodes = extract_unique_nodes(node_a)
    
    assert len(unique_nodes) == 4
    assert set(unique_nodes.keys()) == {"A", "B", "C", "D"}


def test_extract_unique_nodes_with_real_tree(test_tasks_dir):
    """Test extracting unique nodes from real dependency tree (diamond pattern)."""    
    # Build full tree with duplicates
    task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    full_tree = build_full_dependency_tree(task, test_tasks_dir)
    
    # Extract unique nodes
    unique_nodes = extract_unique_nodes(full_tree)
    
    # Should have 4 unique nodes: T1_DIAMOND, T1_DIAMOND_B, T1_DIAMOND_C, T1_LEAF_D
    assert len(unique_nodes) == 4
    expected_tasks = {"T1_DIAMOND", "T1_DIAMOND_B", "T1_DIAMOND_C", "T1_LEAF_D"}
    assert set(unique_nodes.keys()) == expected_tasks
    
    # Verify function names are correct
    assert unique_nodes["T1_DIAMOND"].function_name == "diamond_function"
    assert unique_nodes["T1_DIAMOND_B"].function_name == "diamond_left"
    assert unique_nodes["T1_DIAMOND_C"].function_name == "diamond_right"
    assert unique_nodes["T1_LEAF_D"].function_name == "function_d"


def test_get_dependencies_no_deps(test_tasks_dir):
    """Test getting dependencies for task with no dependencies."""    
    task = load_task(test_tasks_dir / "T1_NO_DEPS.yaml")
    dependencies = get_dependencies(task, test_tasks_dir)
    
    # Should return empty dictionary
    assert dependencies == {}


def test_get_dependencies_single_dependency(test_tasks_dir):
    """Test getting dependencies for task with single dependency."""    
    task = load_task(test_tasks_dir / "T1_CHAIN_B.yaml")  # Depends on T1_LEAF_A
    dependencies = get_dependencies(task, test_tasks_dir)
    
    # Should return only the dependency, not the task itself
    expected = {"T1_LEAF_A": "function_a"}
    assert dependencies == expected


def test_get_dependencies_diamond_pattern(test_tasks_dir):
    """Test getting dependencies for diamond pattern."""    
    task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    dependencies = get_dependencies(task, test_tasks_dir)
    
    # Should return all unique dependencies
    expected = {
        "T1_DIAMOND_B": "diamond_left",
        "T1_DIAMOND_C": "diamond_right",
        "T1_LEAF_D": "function_d"
    }
    assert dependencies == expected
    
    # T1_LEAF_D should appear only once despite being referenced twice
    assert len(dependencies) == 3


@pytest.fixture
def sample_environment():
    """Load sample environment from YAML file."""
    yaml_path = Path(__file__).parent / "example_environments" / "test_environment.yaml"
    return load_environment(yaml_path)


@pytest.fixture
def minimal_environment():
    """Load minimal environment from YAML file."""
    yaml_path = Path(__file__).parent / "example_environments" / "minimal_environment.yaml"
    return load_environment(yaml_path)


def test_full_environment_imports(sample_environment):
    """
    Environment with both required and allowed libraries should return:

        1. required  (numpy, pytest)
        2. allowed   (math, scipy, matplotlib)
        3. typing    (single line)
    """
    imports = build_environment_imports(sample_environment)

    expected = [
        "import numpy as np",
        "import pytest",
        "import math",
        "import scipy as scipy",
        "import matplotlib.pyplot as plt",
        "from typing import List, Dict, Tuple, Callable, Any, Optional, Union",
    ]
    assert imports == expected


def test_minimal_environment_imports(minimal_environment):
    """
    Environment that only lists a single required library still gets the
    typing-helpers line appended.
    """
    imports = build_environment_imports(minimal_environment)

    expected = [
        "import numpy as np",
        "from typing import List, Dict, Tuple, Callable, Any, Optional, Union",
    ]
    assert imports == expected


def test_no_dependencies(test_tasks_dir):
    """Test task with no dependencies - should only return its own code."""
    task = load_task(test_tasks_dir / "T1_NO_DEPS.yaml")
    code_blocks = build_dependency_code_reference(task, test_tasks_dir)
    
    # Should have no dependencies, so empty list
    assert len(code_blocks) == 0
    assert code_blocks == []


def test_single_dependency(test_tasks_dir):
    """Test task with one dependency."""
    task = load_task(test_tasks_dir / "T1_DIAMOND_B.yaml")
    code_blocks = build_dependency_code_reference(task, test_tasks_dir)
    
    # Should have only the dependency code
    assert len(code_blocks) == 1
    
    # Only dependency (T1_LEAF_D)
    assert "def function_d(x: float) -> float:" in code_blocks[0]
    assert "return x / 2.0" in code_blocks[0]


def test_diamond_dependency_pattern(test_tasks_dir):
    """Test complex diamond dependency pattern."""
    task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    code_blocks = build_dependency_code_reference(task, test_tasks_dir)
    
    # Should have 3 blocks: T1_LEAF_D, T1_DIAMOND_B, T1_DIAMOND_C (order may vary)
    assert len(code_blocks) == 3
    
    # Combine all code to check for presence of each function
    all_code = '\n'.join(code_blocks)
    
    # Check that all three dependencies are present
    assert "def function_d(x: float) -> float:" in all_code
    assert "return x / 2.0" in all_code
    
    assert "def diamond_left(x: float) -> float:" in all_code
    assert "return function_d(x) + 1.0" in all_code
    
    assert "def diamond_right(x: float) -> float:" in all_code
    assert "return function_d(x) * 3.0" in all_code


def test_code_blocks_are_separate(test_tasks_dir):
    """Test that each dependency returns separate code blocks."""
    task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    code_blocks = build_dependency_code_reference(task, test_tasks_dir)
    
    # Each block should be a separate string
    for i, block in enumerate(code_blocks):
        assert isinstance(block, str)
        assert len(block.strip()) > 0
        # Each block should only contain one function definition
        assert block.count("def ") == 1


def test_standalone_task_main_code(test_tasks_dir ):
    """Test extracting main code from standalone task."""
    task = load_task(test_tasks_dir / "T1_NO_DEPS.yaml")
    main_code = build_main_code_reference(task)
    
    assert "def standalone_function(x: float) -> float:" in main_code
    assert "return x * 5.0" in main_code


def test_dependency_task_main_code(test_tasks_dir):
    """Test extracting main code from task with dependencies."""
    task = load_task(test_tasks_dir / "T1_DIAMOND_B.yaml")
    main_code = build_main_code_reference(task)
    
    assert "def diamond_left(x: float) -> float:" in main_code
    assert "return function_d(x) + 1.0" in main_code
    # Should not contain dependency code
    assert "def function_d" not in main_code


@pytest.fixture
def test_results_dir():
    """Directory containing example JSON result files."""
    return Path(__file__).parent / "example_results"


def test_build_main_code_llm(test_tasks_dir, test_results_dir):
    """Test extracting main code from LLM JSON output."""
    # Load the task for validation
    task = load_task(test_tasks_dir / "T1_NO_DEPS.yaml")
    
    # Load real JSON file with task validation
    json_file = test_results_dir / "T1_NO_DEPS_output.json"
    parsed_code, errors = validate_llm_output(json_file, task=task)
    
    # Should parse without errors
    assert len(errors) == 0
    assert parsed_code is not None
    
    main_code = build_main_code_llm(parsed_code)
    
    # Check that it matches the YAML reference solution
    assert "def standalone_function(x: float) -> float:" in main_code
    assert "return x * 5.0" in main_code
    assert "Standalone function." in main_code


def test_build_dependency_code_llm_no_dependencies(test_tasks_dir):
    """Test task with no dependencies - should return empty list."""
    task = load_task(test_tasks_dir / "T1_NO_DEPS.yaml")
    llm_outputs = {}
    
    code_blocks = build_dependency_code_llm(task, test_tasks_dir, llm_outputs)
    
    # Should have no dependencies, so empty list
    assert len(code_blocks) == 0


def test_build_dependency_code_llm_diamond_pattern(test_tasks_dir, test_results_dir):
    """Test complex diamond dependency pattern with real JSON files."""
    task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    
    # Load dependency tasks for validation
    leaf_d_task = load_task(test_tasks_dir / "T1_LEAF_D.yaml")
    diamond_b_task = load_task(test_tasks_dir / "T1_DIAMOND_B.yaml")
    diamond_c_task = load_task(test_tasks_dir / "T1_DIAMOND_C.yaml")
    
    # Load real JSON outputs with task validation
    leaf_d_parsed, errors_d = validate_llm_output(
        test_results_dir / "T1_LEAF_D_output.json", task=leaf_d_task
    )
    diamond_b_parsed, errors_b = validate_llm_output(
        test_results_dir / "T1_DIAMOND_B_output.json", task=diamond_b_task
    )
    diamond_c_parsed, errors_c = validate_llm_output(
        test_results_dir / "T1_DIAMOND_C_output.json", task=diamond_c_task
    )
    
    # Should parse without errors
    assert len(errors_d) == 0 and leaf_d_parsed is not None
    assert len(errors_b) == 0 and diamond_b_parsed is not None
    assert len(errors_c) == 0 and diamond_c_parsed is not None
    
    llm_outputs = {
        "T1_LEAF_D": leaf_d_parsed,
        "T1_DIAMOND_B": diamond_b_parsed,
        "T1_DIAMOND_C": diamond_c_parsed
    }
    
    code_blocks = build_dependency_code_llm(task, test_tasks_dir, llm_outputs)
    
    # Should have 3 blocks: T1_LEAF_D, T1_DIAMOND_B, T1_DIAMOND_C (order may vary)
    assert len(code_blocks) == 3
    
    # Combine all code to check for presence of each function
    all_code = '\n'.join(code_blocks)
    
    # Check that all three dependencies are present
    assert "def function_d(x: float) -> float:" in all_code
    assert "return x / 2.0" in all_code
    
    assert "def diamond_left(x: float) -> float:" in all_code
    assert "return function_d(x) + 1.0" in all_code
    
    assert "def diamond_right(x: float) -> float:" in all_code
    assert "return function_d(x) * 3.0" in all_code


def test_aggregate_reference_code_no_deps(test_tasks_dir, sample_environment):
    """Test aggregating code for task with no dependencies."""
    task = load_task(test_tasks_dir / "T1_NO_DEPS.yaml")
    
    aggregated_code = aggregate_reference_code(task, test_tasks_dir, sample_environment)
    
    # Should contain imports
    assert "import numpy as np" in aggregated_code
    assert "import math" in aggregated_code
    
    # Should contain main function
    assert "def standalone_function(x: float) -> float:" in aggregated_code
    assert "return x * 5.0" in aggregated_code
    
    # Should be properly structured (imports first, then function)
    lines = aggregated_code.split('\n')
    import_lines = [i for i, line in enumerate(lines) if line.startswith('import')]
    function_lines = [i for i, line in enumerate(lines) if line.startswith('def')]
    
    # Imports should come before functions
    assert len(import_lines) > 0
    assert len(function_lines) > 0
    assert max(import_lines) < min(function_lines)


def test_aggregate_reference_code_diamond_pattern(test_tasks_dir, sample_environment):
    """Test aggregating code for complex diamond dependency pattern."""
    task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    
    aggregated_code = aggregate_reference_code(task, test_tasks_dir, sample_environment)
    
    # Should contain imports
    assert "import numpy as np" in aggregated_code
    
    # Should contain all dependency functions
    assert "def function_d(x: float) -> float:" in aggregated_code
    assert "def diamond_left(x: float) -> float:" in aggregated_code  
    assert "def diamond_right(x: float) -> float:" in aggregated_code
    
    # Should contain main function
    assert "def diamond_function(x: float) -> float:" in aggregated_code
    assert "return diamond_left(x) * diamond_right(x)" in aggregated_code
    
    # Should be executable Python (basic syntax check)
    try:
        compile(aggregated_code, '<test>', 'exec')
    except SyntaxError:
        pytest.fail("Aggregated code is not valid Python syntax")


def test_aggregate_llm_code_reference_mode(test_tasks_dir, test_results_dir, sample_environment):
    """Test LLM code aggregation using reference dependencies."""
    # Load real files
    task = load_task(test_tasks_dir / "T1_DIAMOND_B.yaml")
    llm_output, errors = validate_llm_output(test_results_dir / "T1_DIAMOND_B_output.json", task=task)
    
    # Should parse without errors
    assert len(errors) == 0 and llm_output is not None
    
    # Aggregate with reference dependencies
    aggregated_code = aggregate_llm_code(
        task, test_tasks_dir, sample_environment, llm_output, DependencyMode.REFERENCE_PROVIDED
    )
    
    # Should contain environment imports, reference dependency, and LLM main function
    assert "import" in aggregated_code  # Has imports
    assert "def function_d(x: float) -> float:" in aggregated_code  # Reference dependency
    assert "def diamond_left(x: float) -> float:" in aggregated_code  # LLM main function
    assert "return function_d(x) + 1.0" in aggregated_code  # LLM implementation
    
    # Should be valid Python
    compile(aggregated_code, '<test>', 'exec')


def test_aggregate_llm_code_llm_chain_mode(test_tasks_dir, test_results_dir, sample_environment):
    """Test LLM code aggregation using LLM chain dependencies."""
    # Load real files
    task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    
    # Load all LLM outputs
    leaf_d_output, _ = validate_llm_output(test_results_dir / "T1_LEAF_D_output.json")
    diamond_b_output, _ = validate_llm_output(test_results_dir / "T1_DIAMOND_B_output.json")
    diamond_c_output, _ = validate_llm_output(test_results_dir / "T1_DIAMOND_C_output.json")
    diamond_main_output, _ = validate_llm_output(test_results_dir / "T1_DIAMOND_output.json", task=task)
    
    llm_outputs = {
        "T1_LEAF_D": leaf_d_output,
        "T1_DIAMOND_B": diamond_b_output,
        "T1_DIAMOND_C": diamond_c_output,
        "T1_DIAMOND_output": diamond_main_output,
    }
    
    # Aggregate with LLM chain dependencies
    aggregated_code = aggregate_llm_code(
        task, test_tasks_dir, sample_environment, diamond_main_output, DependencyMode.LLM_CHAIN, llm_outputs
    )
    
    # Should contain all LLM functions
    assert "def function_d(x: float) -> float:" in aggregated_code
    assert "def diamond_left(x: float) -> float:" in aggregated_code  
    assert "def diamond_right(x: float) -> float:" in aggregated_code
    assert "def diamond_function(x: float) -> float:" in aggregated_code
    
    # Should be valid Python
    compile(aggregated_code, '<test>', 'exec')


def test_aggregate_llm_code_validation_error(test_tasks_dir, test_results_dir, sample_environment):
    """Test error handling for missing llm_outputs in LLM_CHAIN mode."""
    task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    diamond_main_output, _ = validate_llm_output(test_results_dir / "T1_DIAMOND_output.json")
    
    with pytest.raises(ValueError, match="llm_outputs must be provided when dependency_mode is 'llm_chain'"):
        aggregate_llm_code(
            task, test_tasks_dir, sample_environment, diamond_main_output, DependencyMode.LLM_CHAIN, None
        )


def test_evaluate_code_block_success(test_tasks_dir, sample_environment):
    """Test successful evaluation of aggregated reference code."""
    task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    
    # Aggregate the reference code
    code = aggregate_reference_code(task, test_tasks_dir, sample_environment)
    
    # Get function name from task
    function_name = task.expected_function_name
    
    # Test inputs based on task signature
    test_inputs = {"x": 4.0}
    
    # Evaluate the code
    result = evaluate_code_block(code, function_name, test_inputs)
    
    # Assertions
    assert result.success
    assert result.output is not None
    assert isinstance(result.output, (int, float))


def test_evaluate_code_block_missing_function(test_tasks_dir, sample_environment):
    """Test error handling when requested function doesn't exist."""
    task = load_task(test_tasks_dir / "T1_NO_DEPS.yaml")
    
    # Aggregate valid code
    code = aggregate_reference_code(task, test_tasks_dir, sample_environment)
    
    # Try to evaluate a function that doesn't exist
    result = evaluate_code_block(code, "nonexistent_function", {"x": 1.0})
    
    # Should fail gracefully
    assert not result.success
    assert "not found in executed code" in result.error_message
    assert result.output is None


def test_evaluate_code_block_execution_error():
    """Test error handling when function execution fails."""
    # Code that will cause runtime error
    code = """
def division_by_zero(x):
    return x / 0
"""
    
    result = evaluate_code_block(code, "division_by_zero", {"x": 5.0})
    
    # Should fail gracefully
    assert not result.success
    assert "ZeroDivisionError" in result.error_message
    assert result.output is None


# def test_compatible_dtypes_both_numeric():
#     """Test that different numeric types are compatible."""
#     # Different numeric types should be compatible
#     assert _compatible_dtypes(np.dtype('float64'), np.dtype('int32')) == True
#     assert _compatible_dtypes(np.dtype('complex128'), np.dtype('float32')) == True
#     assert _compatible_dtypes(np.dtype('int64'), np.dtype('int16')) == True


# def test_compatible_dtypes_both_non_numeric():
#     """Test that non-numeric types are compatible with each other."""
#     # Both non-numeric should be compatible
#     assert _compatible_dtypes(np.dtype('U10'), np.dtype('U5')) == True  # Unicode strings
#     assert _compatible_dtypes(np.dtype('bool'), np.dtype('bool')) == True
#     assert _compatible_dtypes(np.dtype('object'), np.dtype('object')) == True


# def test_compatible_dtypes_mixed():
#     """Test that numeric and non-numeric types are incompatible."""
#     # Numeric vs non-numeric should be incompatible
#     assert _compatible_dtypes(np.dtype('float64'), np.dtype('U10')) == False
#     assert _compatible_dtypes(np.dtype('int32'), np.dtype('bool')) == False
#     assert _compatible_dtypes(np.dtype('complex128'), np.dtype('object')) == False


# def test_compare_outputs_numeric_within_tolerance():
#     """Test that numeric outputs within tolerance return True."""
#     # Scalars within tolerance
#     assert _compare_outputs(1.0, 1.0000001, 1e-6) == True
#     assert _compare_outputs(1.0, 1.1, 1e-6) == False
    
#     # Arrays within tolerance
#     ref_array = np.array([1.0, 2.0, 3.0])
#     llm_array = np.array([1.0000001, 2.0000001, 3.0000001])
#     assert _compare_outputs(ref_array, llm_array, 1e-6) == True
    
#     # Arrays outside tolerance
#     llm_array_bad = np.array([1.1, 2.1, 3.1])
#     assert _compare_outputs(ref_array, llm_array_bad, 1e-6) == False


def test_compare_outputs_non_numeric():
    """Test that non-numeric outputs use exact equality."""
    # String equality
    assert _compare_outputs("hello", "hello", 1e-6) == True
    assert _compare_outputs("hello", "world", 1e-6) == False
    
    # Boolean equality
    assert _compare_outputs(True, True, 1e-6) == True
    assert _compare_outputs(True, False, 1e-6) == False


# def test_compare_outputs_shape_mismatch():
#     """Test that different shapes return False."""
#     ref_array = np.array([1.0, 2.0])
#     llm_array = np.array([[1.0], [2.0]])  # Different shape
#     assert _compare_outputs(ref_array, llm_array, 1e-6) == False
    
#     # Scalar vs array
#     assert _compare_outputs(1.0, np.array([1.0]), 1e-6) == False


def test_verify_against_reference_all_pass(test_tasks_dir, sample_environment):
    """Test verification when reference and LLM implementations match."""
    # Check if test YAML exists, if not skip or use diamond task
    test_yaml_path = test_tasks_dir / "T1_TEST_VERIFY.yaml"
    if not test_yaml_path.exists():
        pytest.skip("T1_TEST_VERIFY.yaml not found, create it first")
    
    # Load task with multiple test cases
    task = load_task(test_yaml_path)
    
    # Build reference code
    reference_code = aggregate_reference_code(task, test_tasks_dir, sample_environment)
    
    # Create "LLM" code that's identical to reference (should pass all tests)
    llm_code = reference_code  # Perfect match
    
    # Verify
    result = verify_against_reference(task, reference_code, llm_code)
    
    # Should pass all tests
    assert result.all_passed == True
    assert result.total_tests > 0  # At least some tests should exist
    assert result.passed_tests == result.total_tests
    assert len(result.test_results) == result.total_tests
    
    # Check individual test results
    for test_result in result.test_results:
        assert test_result['passed'] == True
        assert test_result['error'] is None


def test_verify_against_reference_some_fail(test_tasks_dir, sample_environment):
    """Test verification when LLM implementation has errors."""
    # Load task
    task = load_task(test_tasks_dir / "T1_TEST_VERIFY.yaml")
    
    # Build reference code
    reference_code = aggregate_reference_code(task, test_tasks_dir, sample_environment)
    
    # Create LLM code with wrong implementation
    llm_code = """
import numpy as np

def linear_function(x: float) -> float:
    # Wrong implementation: f(x) = 3*x + 2 instead of 2*x + 1
    return 3.0 * x + 2.0
"""
    
    # Verify
    result = verify_against_reference(task, reference_code, llm_code)
    
    # Should fail all tests (wrong formula)
    assert result.all_passed == False
    assert result.total_tests == 4
    assert result.passed_tests == 0  # All should fail
    
    # Check that failures are detected
    for test_result in result.test_results:
        assert test_result['passed'] == False
        assert "differ beyond tolerance" in test_result['error']
        assert test_result['reference_output'] != test_result['llm_output']


def test_verify_against_reference_llm_error(test_tasks_dir, sample_environment):
    """Test verification when LLM code has execution errors."""
    # Load task
    task = load_task(test_tasks_dir / "T1_TEST_VERIFY.yaml")
    
    # Build reference code
    reference_code = aggregate_reference_code(task, test_tasks_dir, sample_environment)
    
    # Create LLM code that will cause runtime error
    llm_code = """
import numpy as np

def linear_function(x: float) -> float:
    # This will cause ZeroDivisionError
    return x / 0.0
"""
    
    # Verify
    result = verify_against_reference(task, reference_code, llm_code)
    
    # Should fail all tests due to execution errors
    assert result.all_passed == False
    assert result.total_tests == 4
    assert result.passed_tests == 0
    
    # Check that execution errors are captured
    for test_result in result.test_results:
        assert test_result['passed'] == False
        assert "LLM evaluation failed" in test_result['error']
        assert "ZeroDivisionError" in test_result['error']
        assert test_result['llm_output'] is None


# Alternative test using the diamond task if T1_TEST_VERIFY doesn't exist
def test_verify_diamond_task(test_tasks_dir, sample_environment):
    """Test verification using existing diamond task."""
    try:
        # Try to load diamond task
        task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    except FileNotFoundError:
        pytest.skip("T1_DIAMOND.yaml not found")
    
    # Check if task has test cases for verification
    test_cases = getattr(task, 'test_cases', [])
    if not test_cases:
        pytest.skip("Diamond task has no reference verification test cases")
    
    # Build reference code
    reference_code = aggregate_reference_code(task, test_tasks_dir, sample_environment)
    
    # Use same code as LLM (should pass)
    llm_code = reference_code
    
    # Verify
    result = verify_against_reference(task, reference_code, llm_code)
    
    # Should pass
    assert result.all_passed == True
    assert result.passed_tests == result.total_tests
    assert result.total_tests == len(test_cases)


def test_evaluate_leaf_task(test_tasks_dir, test_results_dir, sample_environment):
    """Test evaluate_task_implementation with a leaf task (no dependencies)."""
    # Check if leaf task exists
    leaf_yaml_path = test_tasks_dir / "T1_LEAF_D.yaml"
    if not leaf_yaml_path.exists():
        pytest.skip("T1_LEAF_D.yaml not found")
    
    # Load and parse the LLM output from test_results_dir
    leaf_output_path = test_results_dir / "T1_LEAF_D_output.json"
    if not leaf_output_path.exists():
        pytest.skip("T1_LEAF_D_output.json not found")
    
    # Load task for validation
    task = load_task(leaf_yaml_path)
    
    # Parse and validate JSON to ParsedCode object
    parsed_llm_output, validation_errors = validate_llm_output(leaf_output_path, task)
    
    # Check that parsing succeeded
    assert parsed_llm_output is not None, f"Failed to parse LLM output: {validation_errors}"
    
    # Test with REFERENCE_PROVIDED mode (should work same as LLM_CHAIN for leaf tasks)
    result = evaluate_task_implementation(
        task_yaml_path=leaf_yaml_path,
        llm_output=parsed_llm_output,
        tasks_directory=test_tasks_dir,
        environment=sample_environment,
        dependency_mode=DependencyMode.REFERENCE_PROVIDED
    )
    
    # Verify the evaluation succeeded
    assert isinstance(result, TaskEvaluationResult)
    assert result.success == True
    assert result.error_message is None
    assert result.task_id == "T1_LEAF_D"
    
    # Verify code blocks were constructed
    assert len(result.reference_code) > 0
    assert len(result.llm_code) > 0
    assert "function_d" in result.llm_code
    
    # Verify verification was performed
    assert result.verification_result is not None
    
    # For the specific test case: function_d(4.0) should equal 2.0
    # Reference: x / 2.0, so 4.0 / 2.0 = 2.0
    if result.verification_result.total_tests > 0:
        # Should pass since LLM output matches reference
        assert result.verification_result.all_passed == True


def test_evaluate_diamond_reference_provided(test_tasks_dir, test_results_dir, sample_environment):
    """Test evaluate_task_implementation with diamond task using REFERENCE_PROVIDED mode."""
    # Check if diamond task exists
    diamond_yaml_path = test_tasks_dir / "T1_DIAMOND.yaml"
    if not diamond_yaml_path.exists():
        pytest.skip("T1_DIAMOND.yaml not found")
    
    # Load and parse the actual LLM output from the JSON file in test_results_dir
    diamond_output_path = test_results_dir / "T1_DIAMOND_output.json"
    if not diamond_output_path.exists():
        pytest.skip("T1_DIAMOND_output.json not found")
    
    # Load task for validation
    task = load_task(diamond_yaml_path)
    
    # Parse and validate JSON to ParsedCode object
    parsed_llm_output, validation_errors = validate_llm_output(diamond_output_path, task)
    
    # Check that parsing succeeded
    assert parsed_llm_output is not None, f"Failed to parse LLM output: {validation_errors}"
    
    # Optionally check for validation errors (might be warnings, not failures)
    if validation_errors:
        print(f"Validation warnings for T1_DIAMOND: {validation_errors}")
    
    # Evaluate using REFERENCE_PROVIDED mode
    result = evaluate_task_implementation(
        task_yaml_path=diamond_yaml_path,
        llm_output=parsed_llm_output,
        tasks_directory=test_tasks_dir,
        environment=sample_environment,
        dependency_mode=DependencyMode.REFERENCE_PROVIDED
    )
    
    # Verify the evaluation succeeded
    assert isinstance(result, TaskEvaluationResult)
    assert result.success == True
    assert result.error_message is None
    assert result.dependency_mode == DependencyMode.REFERENCE_PROVIDED
    assert result.task_id == "T1_DIAMOND"
    
    # Verify code blocks were constructed
    assert len(result.reference_code) > 0
    assert len(result.llm_code) > 0
    assert "import numpy as np" in result.reference_code
    assert "import numpy as np" in result.llm_code
    
    # Should include dependencies in REFERENCE_PROVIDED mode
    assert "diamond_left" in result.reference_code
    assert "diamond_right" in result.reference_code
    assert "function_d" in result.reference_code
    
    # Verify verification was performed
    assert result.verification_result is not None
    assert result.verification_result.total_tests >= 0


def test_evaluate_diamond_llm_chain(test_tasks_dir, test_results_dir, sample_environment):
    """Test evaluate_task_implementation with diamond task using LLM_CHAIN mode."""
    # Check if diamond task exists
    diamond_yaml_path = test_tasks_dir / "T1_DIAMOND.yaml"
    if not diamond_yaml_path.exists():
        pytest.skip("T1_DIAMOND.yaml not found")
    
    # Load all the LLM outputs for the chain from test_results_dir
    output_files = [
        "T1_DIAMOND_output.json",
        "T1_DIAMOND_B_output.json", 
        "T1_DIAMOND_C_output.json",
        "T1_LEAF_D_output.json"
    ]
    
    # Check all files exist
    for output_file in output_files:
        if not (test_results_dir / output_file).exists():
            pytest.skip(f"{output_file} not found in test_results_dir")
    
    # Load tasks for validation
    diamond_task = load_task(diamond_yaml_path)
    diamond_b_task = load_task(test_tasks_dir / "T1_DIAMOND_B.yaml")
    diamond_c_task = load_task(test_tasks_dir / "T1_DIAMOND_C.yaml")
    leaf_d_task = load_task(test_tasks_dir / "T1_LEAF_D.yaml")
    
    # Parse and validate all LLM outputs
    parsed_diamond, diamond_errors = validate_llm_output(test_results_dir / "T1_DIAMOND_output.json", diamond_task)
    parsed_diamond_b, diamond_b_errors = validate_llm_output(test_results_dir / "T1_DIAMOND_B_output.json", diamond_b_task)
    parsed_diamond_c, diamond_c_errors = validate_llm_output(test_results_dir / "T1_DIAMOND_C_output.json", diamond_c_task)
    parsed_leaf_d, leaf_d_errors = validate_llm_output(test_results_dir / "T1_LEAF_D_output.json", leaf_d_task)
    
    # Check that all parsing succeeded
    assert parsed_diamond is not None, f"Failed to parse T1_DIAMOND: {diamond_errors}"
    assert parsed_diamond_b is not None, f"Failed to parse T1_DIAMOND_B: {diamond_b_errors}"
    assert parsed_diamond_c is not None, f"Failed to parse T1_DIAMOND_C: {diamond_c_errors}"
    assert parsed_leaf_d is not None, f"Failed to parse T1_LEAF_D: {leaf_d_errors}"
    
    # Create llm_outputs dict for LLM_CHAIN mode
    llm_outputs = {
        "T1_DIAMOND_B": parsed_diamond_b,
        "T1_DIAMOND_C": parsed_diamond_c,
        "T1_LEAF_D": parsed_leaf_d
    }
    
    # Evaluate using LLM_CHAIN mode
    result = evaluate_task_implementation(
        task_yaml_path=diamond_yaml_path,
        llm_output=parsed_diamond,
        tasks_directory=test_tasks_dir,
        environment=sample_environment,
        dependency_mode=DependencyMode.LLM_CHAIN,
        llm_outputs=llm_outputs
    )
    
    # Verify the evaluation succeeded
    assert isinstance(result, TaskEvaluationResult)
    assert result.success == True
    assert result.error_message is None
    assert result.dependency_mode == DependencyMode.LLM_CHAIN
    assert result.task_id == "T1_DIAMOND"
    
    # Verify code blocks were constructed
    assert len(result.reference_code) > 0
    assert len(result.llm_code) > 0
    
    # In LLM_CHAIN mode, should use LLM implementations for dependencies
    assert "diamond_function" in result.llm_code
    
    # Verify verification was performed
    assert result.verification_result is not None
    assert result.verification_result.total_tests >= 0


@pytest.fixture
def test_tasks_dir():
    return Path(__file__).parent / "example_tasks"


@pytest.fixture
def test_results_dir():
    return Path(__file__).parent / "example_results"


@pytest.fixture
def test_env_dir():
    return Path(__file__).parent / "example_environments"


@pytest.fixture
def minimal_env(test_env_dir):
    """Load the minimal environment YAML."""
    return load_environment(test_env_dir / "minimal_environment.yaml")


@pytest.fixture
def parsed_llm_output(test_tasks_dir, test_results_dir):
    """Load and parse the LLM JSON output for T1_NO_DEPS."""
    task = load_task(test_tasks_dir / "T1_NO_DEPS.yaml")
    parsed_code, errors = validate_llm_output(
        test_results_dir / "T1_NO_DEPS_output.json",
        task=task
    )
    assert not errors, f"Validation errors: {errors}"
    return parsed_code


@pytest.fixture
def implementation_code(parsed_llm_output):
    """Extract main function code from parsed LLM output."""
    return parsed_llm_output.main_function


@pytest.fixture
def test_functions(parsed_llm_output):
    """Extract test functions dictionary from parsed LLM output."""
    return parsed_llm_output.test_functions


def test_test_evaluator_success(minimal_env, implementation_code, test_functions):
    evaluator = TestEvaluator(minimal_env)
    results = evaluator.run_tests_against_implementation(test_functions, implementation_code)

    assert results, "No test results returned"
    for result in results.values():
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.error is None


def test_llm_tests_pass_on_reference_T1_DIAMOND(test_tasks_dir, test_results_dir, minimal_environment):
    from fem_bench.prompt import validate_llm_output
    from fem_bench.evaluate import (
        TestEvaluator,
        build_dependency_code_reference,
        build_environment_imports
    )

    # Load T1_DIAMOND task and its LLM output
    task = load_task(test_tasks_dir / "T1_DIAMOND.yaml")
    parsed, errors = validate_llm_output(test_results_dir / "T1_DIAMOND_output.json", task)
    assert not errors, f"LLM output invalid: {errors}"

    # Build the reference implementation (imports + dependencies + main)
    imports = build_environment_imports(minimal_environment)
    import_code = "\n".join(imports)

    dep_code_blocks = build_dependency_code_reference(task, test_tasks_dir)
    main_code = task.reference_solution.code

    full_reference_code = "\n\n".join([import_code] + dep_code_blocks + [main_code])

    # Run LLM test functions against the reference code
    evaluator = TestEvaluator(minimal_environment)
    results = evaluator.run_tests_against_implementation(
        test_functions=parsed.test_functions,
        implementation_code=full_reference_code
    )

    # Check that all tests pass
    assert results, "No LLM test functions found in parsed output"
    for name, result in results.items():
        assert result.passed, f"LLM test {name} failed on reference: {result.error}"


def test_test_quality_evaluator_with_expected_failures(test_tasks_dir, test_results_dir, minimal_environment):
    # Load task and LLM output
    task_path = test_tasks_dir / "T1_FAILURE_EVAL.yaml"
    json_path = test_results_dir / "T1_FAILURE_EVAL_output.json"

    task = load_task(task_path)
    parsed, errors = validate_llm_output(json_path, task)
    assert not errors, f"LLM output validation failed: {errors}"

    evaluator = TestQualityEvaluator(minimal_environment)

    # Test against reference code (LLM tests should pass)
    ref_result: ReferenceTestResult = evaluator.test_against_reference(task, parsed)
    assert ref_result.all_passed, "LLM-generated test failed on reference implementation"
    for name, result in ref_result.test_results.items():
        assert result.passed, f"{name} failed on reference: {result.error}"

    # Test against buggy code (LLM tests should fail as expected)
    fail_result: FailureDetectionResult = evaluator.run_tests_against_failures(task, parsed)
    assert fail_result.all_expected_caught, "Some expected failures were not caught by LLM tests"

    # Optional debug print
    for missed in fail_result.expected_failures_missed:
        print(f"Missed failure: {missed.failure_type} via {missed.test_function}")
        assert False, f"LLM test failed to detect known bug: {missed.failure_type}"


@pytest.fixture
def test_files():
    """Path to the test files using relative directory structure."""
    test_dir = Path(__file__).parent  # Directory where this test file is located
    
    return {
        "task_file": test_dir / "example_tasks" / "TEST_SIMPLE.yaml",
        "good_output_file": test_dir / "example_results" / "TEST_SIMPLE_good.json",
        "bad_output_file": test_dir / "example_results" / "TEST_SIMPLE_bad.json",
        "env_file": test_dir / "example_environments" / "simple_environment.yaml",
        "task_dir": test_dir / "example_tasks"  # For dependency resolution
    }


def test_evaluate_good_llm_output(test_files):
    """Test with good LLM output - all tests should pass."""
    
    # Load files directly
    task = load_task(test_files["task_file"])
    environment = load_environment(test_files["env_file"])
    llm_output, validation_errors = validate_llm_output(test_files["good_output_file"], task)
    
    # Verify parsing worked
    assert len(validation_errors) == 0
    assert llm_output.main_function_name == "add_ten"
    assert len(llm_output.test_functions) == 2
    
    # Evaluate
    result = evaluate_single_task(
        task=task,
        environment=environment,
        llm_output=llm_output,
        task_dir=test_files["task_dir"],
        llm_outputs={}  # No dependencies
    )
    
    # Verify results
    assert result["task_id"] == "TEST_SIMPLE"
    assert result["total_tests"] == 2
    assert result["tests_passed_on_reference"] == 2  # Both tests should pass
    assert result["total_expected_failures"] == 2  # Two expected failures defined
    assert result["expected_failures_failed_on_reference"] == 2  # Should catch both
    assert result["fcn_correct_with_reference_provided"] == 1  # Function correct
    assert result["fcn_correct_with_llm_chain"] == 1  # No deps, so same result


def test_evaluate_bad_llm_output(test_files):
    """Test with bad LLM output - tests should fail on reference."""
    
    # Load files directly
    task = load_task(test_files["task_file"])
    environment = load_environment(test_files["env_file"])
    llm_output, validation_errors = validate_llm_output(test_files["bad_output_file"], task)
    
    # Verify parsing worked
    assert len(validation_errors) == 0
    assert llm_output.main_function_name == "add_ten"
    
    # Evaluate
    result = evaluate_single_task(
        task=task,
        environment=environment,
        llm_output=llm_output,
        task_dir=test_files["task_dir"],
        llm_outputs={}
    )
    
    # Verify results
    assert result["task_id"] == "TEST_SIMPLE"
    assert result["total_tests"] == 2
    assert result["tests_passed_on_reference"] == 0  # Both tests should fail (wrong assertions)
    assert result["total_expected_failures"] == 2
    # Note: expected_failures_failed_on_reference might be 0 since tests themselves are wrong
    assert result["fcn_correct_with_reference_provided"] == 1  # Function is still correct
    assert result["fcn_correct_with_llm_chain"] == 1


def test_failure_detection(test_files):
    """Test that failure detection works correctly."""
    
    # Load files directly
    task = load_task(test_files["task_file"])
    environment = load_environment(test_files["env_file"])
    llm_output, _ = validate_llm_output(test_files["good_output_file"], task)
    
    # Evaluate
    result = evaluate_single_task(
        task=task,
        environment=environment,
        llm_output=llm_output,
        task_dir=test_files["task_dir"],
        llm_outputs={}
    )
    
    # The good tests should catch both failure examples
    # - test_add_ten_basic should catch "wrong_operation" (x - 10 instead of x + 10)
    # - test_add_ten_negative should catch "off_by_one" (x + 11 instead of x + 10)
    assert result["expected_failures_failed_on_reference"] == 2
    
    # All reference tests should pass
    assert result["tests_passed_on_reference"] == 2
    
    # Function implementation should be correct
    assert result["fcn_correct_with_reference_provided"] == 1
