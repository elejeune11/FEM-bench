from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
from typing import Any, Dict, Set, Optional, List, Union
from fem_bench.yaml_load import load_task, Task, ExpectedFailure
from fem_bench.yaml_load import Environment, Library
from fem_bench.prompt import parse_llm_json_output, ParsedCode


class DependencyError(Exception):
    """Raised when a dependency cannot be resolved."""
    pass


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""
    pass


@dataclass
class DependencyNode:
    """Represents a single node in the dependency tree."""
    task_id: str
    function_name: str
    dependencies: List['DependencyNode']
    task: Optional[Task] = None


def build_full_dependency_tree(
    task: Task,
    task_dir: Path,
    resolution_stack: Optional[Set[str]] = None
) -> DependencyNode:
    """
    Build complete dependency tree with potential duplicates.
    
    Args:
        task: Task to build tree for
        task_dir: Directory containing task YAML files
        resolution_stack: Stack for circular dependency detection
        
    Returns:
        Root DependencyNode with full tree structure
        
    Raises:
        CircularDependencyError: If circular dependencies detected
        DependencyError: If dependency task YAML file not found
    """
    if resolution_stack is None:
        resolution_stack = set()
    
    # Check for circular dependencies
    if task.task_id in resolution_stack:
        raise CircularDependencyError(
            f"Circular dependency detected: {' -> '.join(resolution_stack)} -> {task.task_id}"
        )
    
    resolution_stack.add(task.task_id)
    
    try:
        # Process all dependencies recursively
        dependency_nodes = []
        for dep in task.task_dependencies:
            dep_task_id = dep.source_task
            
            # Load dependency task
            dep_task_path = task_dir / f"{dep_task_id}.yaml"
            if not dep_task_path.exists():
                raise DependencyError(f"Dependency task file not found: {dep_task_path}")
            
            dep_task = load_task(dep_task_path)
            
            # Recursively build subtree (may contain duplicates)
            dep_node = build_full_dependency_tree(dep_task, task_dir, resolution_stack)
            dependency_nodes.append(dep_node)
        
        # Create node for current task
        return DependencyNode(
            task_id=task.task_id,
            function_name=task.expected_function_name,
            dependencies=dependency_nodes,
            task=task
        )
    
    finally:
        resolution_stack.discard(task.task_id)


def extract_unique_nodes(root_node: DependencyNode) -> Dict[str, DependencyNode]:
    """
    Extract all unique nodes from a dependency tree.
    
    Args:
        root_node: Root of the dependency tree (may contain duplicates)
        
    Returns:
        Dictionary mapping task_id -> DependencyNode (first occurrence wins)
        
    Example:
        In a diamond pattern where T1_LEAF_D appears twice, this function
        will return only one T1_LEAF_D node in the result dictionary.
    """
    unique_nodes = {}
    
    def collect_node(node: DependencyNode):
        """Recursively collect unique nodes (first occurrence wins)."""
        # Add this node if we haven't seen it before
        if node.task_id not in unique_nodes:
            unique_nodes[node.task_id] = node
        
        # Recursively process all dependencies
        for dep_node in node.dependencies:
            collect_node(dep_node)
    
    collect_node(root_node)
    return unique_nodes


def get_dependencies(task, task_dir: Path) -> Dict[str, str]:
    """
    Get all dependencies for a task.
    
    Args:
        task: Task object (loaded from YAML)
        task_dir: Directory containing task YAML files
        
    Returns:
        Dictionary mapping dependency task_id -> function_name
        (excludes the original task itself)
        
    Example:
        For a diamond pattern T1_DIAMOND → T1_DIAMOND_B, T1_DIAMOND_C; 
        T1_DIAMOND_B → T1_LEAF_D; T1_DIAMOND_C → T1_LEAF_D
        
        Returns: {
            "T1_DIAMOND_B": "diamond_left",
            "T1_DIAMOND_C": "diamond_right", 
            "T1_LEAF_D": "function_d"
        }
    """
    # Step 1: Build full dependency tree (with duplicates)
    full_tree = build_full_dependency_tree(task, task_dir)
    
    # Step 2: Extract unique nodes
    unique_nodes = extract_unique_nodes(full_tree)
    
    # Step 3: Convert to task_id -> function_name mapping, excluding root task
    dependencies = {}
    for task_id, node in unique_nodes.items():
        if task_id != task.task_id:  # Exclude the original task
            dependencies[task_id] = node.function_name
    
    return dependencies


def build_environment_imports(environment: Environment) -> List[str]:
    """
    Build import statements based on an Environment object.

    Order:
        1. Required libraries       (preserves yaml order)
        2. Allowed libraries        (preserves yaml order)
        3. Typing import (single line)

    Returns
    -------
    List[str]
        Fully-qualified import lines, ready to ``'\n'.join(imports)``.
    """
    imports: List[str] = []

    # ------------------------------------------------------------------ #
    # 1) required                                                         #
    # ------------------------------------------------------------------ #
    for lib in environment.required_libraries:
        if lib.import_as:
            imports.append(f"import {lib.name} as {lib.import_as}")
        else:
            imports.append(f"import {lib.name}")

    # ------------------------------------------------------------------ #
    # 2) allowed                                                          #
    # ------------------------------------------------------------------ #
    for lib in environment.allowed_libraries:
        if lib.import_as:
            imports.append(f"import {lib.name} as {lib.import_as}")
        else:
            imports.append(f"import {lib.name}")

    # ------------------------------------------------------------------ #
    # 3) typing helpers                                                   #
    # ------------------------------------------------------------------ #
    typing_line: str | None = None
    for line in environment.import_guidelines.splitlines():
        stripped = line.strip()
        if stripped.startswith("from typing import"):
            typing_line = stripped
            break

    if typing_line is None:
        # Fallback – cover the common aliases you asked for
        typing_line = (
            "from typing import List, Dict, Tuple, Callable, "
            "Any, Optional, Union"
        )

    # Avoid duplicates if caller added it manually earlier
    if typing_line not in imports:
        imports.append(typing_line)

    return imports


def build_dependency_code_reference(task: Task, task_dir: Path) -> List[str]:
    """
    Build dependency code from task reference solutions using dependency tree resolution.
    
    Args:
        task: Task object containing dependency specifications
        task_dir: Directory containing all task YAML files
        
    Returns:
        List of dependency code strings, in dependency order.
        Does NOT include the main task's code.
        Handles complex dependency patterns including diamond dependencies.
        
    Examples:
        >>> task = load_task("task_with_deps.yaml")
        >>> code_blocks = build_dependency_code_reference(task, task_dir)
        >>> print('\n\n'.join(code_blocks))
        def dependency_function():
            return np.array([1, 2, 3])
    """
    # Get all unique dependencies using the dependency tree
    dependencies = get_dependencies(task, task_dir)
    
    code_blocks = []
    
    # Extract reference code for each dependency in order
    for task_id, function_name in dependencies.items():
        dep_task_path = task_dir / f"{task_id}.yaml"
        dep_task = load_task(dep_task_path)
        code_blocks.append(dep_task.reference_solution.code)
    
    return code_blocks


def build_main_code_reference(task: Task) -> str:
    """
    Build main code from task reference solution.
    
    Args:
        task: Task object containing reference solution
        
    Returns:
        Main task implementation code as a string
        
    Examples:
        >>> task = load_task("task.yaml")
        >>> main_code = build_main_code_reference(task)
        >>> print(main_code)
        def main_function(x: float) -> float:
            return x * 2.0
    """
    return task.reference_solution.code


def build_main_code_llm(llm_output: ParsedCode) -> str:
    """
    Build main code from LLM JSON output.
    
    Args:
        llm_output: ParsedCode object from parse_llm_json_output
        
    Returns:
        Main task implementation code as a string
        
    Examples:
        >>> parsed = parse_llm_json_output(json_data)
        >>> main_code = build_main_code_llm(parsed)
        >>> print(main_code)
        def main_function(x: float) -> float:
            return x * 2.0
    """
    return llm_output.main_function


def build_dependency_code_llm(task: Task, task_dir: Path, llm_outputs: Dict[str, ParsedCode]) -> List[str]:
    """
    Build dependency code from LLM JSON outputs using dependency tree resolution.
    
    Args:
        task: Task object containing dependency specifications
        task_dir: Directory containing all task YAML files
        llm_outputs: Dict mapping task_id -> ParsedCode for dependencies
        
    Returns:
        List of dependency code strings, in dependency order.
        Does NOT include the main task's code.
        Handles complex dependency patterns including diamond dependencies.
        
    Examples:
        >>> llm_outputs = {"T1_LEAF_D": parsed_leaf_d, "T1_DIAMOND_B": parsed_diamond_b}
        >>> code_blocks = build_dependency_code_llm(task, task_dir, llm_outputs)
        >>> print('\n\n'.join(code_blocks))
        def function_d(x: float) -> float:
            return x / 2.0
        
        def diamond_left(x: float) -> float:
            return function_d(x) + 1.0
    """
    # Get all unique dependencies using the dependency tree
    dependencies = get_dependencies(task, task_dir)
    
    code_blocks = []
    
    # Extract code for each dependency in order
    for task_id, function_name in dependencies.items():
        if task_id not in llm_outputs:
            raise ValueError(f"Missing LLM output for dependency: {task_id}")
        
        llm_output = llm_outputs[task_id]
        code_blocks.append(llm_output.main_function)
    
    return code_blocks


def aggregate_reference_code(task: Task, task_dir: Path, environment: Environment) -> str:
    """
    Aggregate complete reference implementation from imports, dependencies, and main function.
    
    Args:
        task: Task object containing the main function specification
        task_dir: Directory containing all task YAML files
        environment: Environment object containing import specifications
        
    Returns:
        Complete Python code string ready for execution
        
    Examples:
        >>> task = load_task("task.yaml")
        >>> env = load_environment("env.yaml")
        >>> code = aggregate_reference_code(task, task_dir, env)
        >>> print(code)
        import numpy as np
        import math
        
        def dependency_function(x: float) -> float:
            return x / 2.0
        
        def main_function(x: float) -> float:
            return dependency_function(x) * 2.0
    """
    # Get all components
    imports = build_environment_imports(environment)
    dependency_code = build_dependency_code_reference(task, task_dir)
    main_code = build_main_code_reference(task)
    
    # Combine into sections
    sections = []
    
    # Add imports section
    if imports:
        import_section = '\n'.join(imports)
        sections.append(import_section)
    
    # Add dependency code section
    if dependency_code:
        code_section = '\n\n'.join(dependency_code)
        sections.append(code_section)
    
    # Add main code section
    sections.append(main_code)
    
    # Join all sections with double newlines
    return '\n\n'.join(sections)


class DependencyMode(Enum):
    """Dependency handling modes for LLM code aggregation."""
    REFERENCE_PROVIDED = "reference_provided"
    LLM_CHAIN = "llm_chain"


def aggregate_llm_code(
    task: Task,
    task_dir: Path,
    environment: Environment,
    llm_output: ParsedCode,
    dependency_mode: DependencyMode,
    llm_outputs: Dict[str, ParsedCode] = None
) -> str:
    """
    Aggregate complete LLM implementation with different dependency handling modes.
    
    Args:
        task: Task object containing the main function specification
        task_dir: Directory containing all task YAML files
        environment: Environment object containing import specifications
        llm_output: ParsedCode object for the main task
        dependency_mode: How to handle dependencies
            - "reference_provided": Use reference implementations
            - "llm_chain": Use LLM implementations from previous tasks
        llm_outputs: Dict mapping task_id -> ParsedCode (required for "llm_chain" mode)
        
    Returns:
        Complete Python code string ready for execution
        
    Raises:
        ValueError: If llm_outputs is None when dependency_mode is "llm_chain"
        
    Examples:
        # Reference provided mode
        >>> code = aggregate_llm_code(task, task_dir, env, llm_out, DependencyMode.REFERENCE_PROVIDED)
        
        # LLM chain mode  
        >>> llm_outs = {"T1_LEAF_D": parsed_d, "T1_DIAMOND_B": parsed_b}
        >>> code = aggregate_llm_code(task, task_dir, env, llm_out, DependencyMode.LLM_CHAIN, llm_outs)
    """
    # Validate inputs for llm_chain mode
    if dependency_mode == DependencyMode.LLM_CHAIN and llm_outputs is None:
        raise ValueError("llm_outputs must be provided when dependency_mode is 'llm_chain'")
    
    # Get imports (same for both modes)
    imports = build_environment_imports(environment)
    
    # Get dependency code based on mode
    if dependency_mode == DependencyMode.REFERENCE_PROVIDED:
        dependency_code = build_dependency_code_reference(task, task_dir)
    elif dependency_mode == DependencyMode.LLM_CHAIN:
        dependency_code = build_dependency_code_llm(task, task_dir, llm_outputs)
    else:
        raise ValueError(f"Invalid dependency_mode: {dependency_mode}")
    
    # Get main function code from LLM output
    main_code = build_main_code_llm(llm_output)
    
    # Combine into sections
    sections = []
    
    # Add imports section
    if imports:
        import_section = '\n'.join(imports)
        sections.append(import_section)
    
    # Add dependency code section
    if dependency_code:
        code_section = '\n\n'.join(dependency_code)
        sections.append(code_section)
    
    # Add main code section
    sections.append(main_code)
    
    # Join all sections with double newlines
    return '\n\n'.join(sections)


@dataclass
class EvaluationResult:
    """Result of code evaluation."""
    success: bool
    output: Any = None
    error_message: str = None


def evaluate_code_block(
    code: str,
    function_name: str,
    test_inputs: Dict[str, Any]
) -> EvaluationResult:
    """
    Evaluate a code block by executing it and calling the specified function.
    
    Args:
        code: Complete Python code string (output from aggregate_*_code functions)
        function_name: Name of the function to evaluate
        test_inputs: Dictionary of parameter_name -> value for function call
        
    Returns:
        EvaluationResult containing success status, output, and any error info
        
    Examples:
        >>> code = aggregate_reference_code(task, task_dir, env)
        >>> inputs = {"x": 2.0, "y": 3.0}
        >>> result = evaluate_code_block(code, "my_function", inputs)
        >>> if result.success:
        ...     print(f"Function returned: {result.output}")
    """
    try:
        # Create execution namespace with numpy
        namespace = {'__builtins__': __builtins__}
        
        # Execute the code block
        exec(code, namespace)
        
        # Check if function exists
        if function_name not in namespace:
            return EvaluationResult(
                success=False,
                error_message=f"Function '{function_name}' not found in executed code"
            )
        
        target_function = namespace[function_name]
        
        # Call the function with test inputs as keyword arguments
        result = target_function(**test_inputs)
        
        return EvaluationResult(
            success=True,
            output=result
        )
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        
        return EvaluationResult(
            success=False,
            error_message=error_msg
        )


@dataclass
class VerificationResult:
    """Result of reference vs LLM verification."""
    all_passed: bool
    test_results: List[Dict[str, Any]]
    total_tests: int
    passed_tests: int


def verify_against_reference(
    task: Any,
    reference_code: str,
    llm_code: str
) -> VerificationResult:
    """
    Compare LLM implementation against reference implementation using task test cases.
    
    Args:
        task: Task object loaded from YAML (contains reference_verification section)
        reference_code: Complete reference code string (from aggregate_reference_code)
        llm_code: Complete LLM code string (from aggregate_llm_code)
        
    Returns:
        VerificationResult indicating if all tests passed and detailed results
        
    Examples:
        >>> task = load_task("T1_SF_001.yaml")
        >>> ref_code = aggregate_reference_code(task, task_dir, env)
        >>> llm_code = aggregate_llm_code(task, task_dir, env, llm_output, mode)
        >>> result = verify_against_reference(task, ref_code, llm_code)
        >>> if result.all_passed:
        ...     print(f"All {result.total_tests} tests passed!")
    """
    # Extract function name and test cases from task object
    function_name = task.expected_function_name
    
    # Get test cases from the task object
    test_cases = getattr(task, 'test_cases', [])
    
    if not test_cases:
        # No test cases defined - return success by default
        return VerificationResult(
            all_passed=True,
            test_results=[],
            total_tests=0,
            passed_tests=0
        )
    
    test_results = []
    passed_count = 0
    
    for i, test_case in enumerate(test_cases):
        # Access test case attributes directly
        test_input = test_case.input
        tolerance = test_case.tolerance
        
        # Evaluate reference implementation
        ref_result = evaluate_code_block(reference_code, function_name, test_input)
        
        # Evaluate LLM implementation  
        llm_result = evaluate_code_block(llm_code, function_name, test_input)
        
        print("RESULT GROUP:")
        print(ref_result)
        print(llm_result)

        # Check if both evaluations succeeded
        if not ref_result.success:
            test_results.append({
                'test_index': i,
                'input': test_input,
                'tolerance': tolerance,
                'passed': False,
                'error': f"Reference evaluation failed: {ref_result.error_message}",
                'reference_output': None,
                'llm_output': None
            })
            continue
            
        if not llm_result.success:
            test_results.append({
                'test_index': i,
                'input': test_input,
                'tolerance': tolerance,
                'passed': False,
                'error': f"LLM evaluation failed: {llm_result.error_message}",
                'reference_output': ref_result.output,
                'llm_output': None
            })
            continue
        
        # Compare outputs within tolerance
        outputs_match = _compare_outputs(
            ref_result.output, 
            llm_result.output, 
            tolerance
        )
        print(outputs_match)
        
        if outputs_match:
            passed_count += 1
        
        test_results.append({
            'test_index': i,
            'input': test_input,
            'tolerance': tolerance,
            'passed': outputs_match,
            'error': None if outputs_match else f"Outputs differ beyond tolerance {tolerance}",
            'reference_output': ref_result.output,
            'llm_output': llm_result.output
        })
    
    all_passed = passed_count == len(test_cases)
    
    return VerificationResult(
        all_passed=all_passed,
        test_results=test_results,
        total_tests=len(test_cases),
        passed_tests=passed_count
    )


def _compare_outputs(ref_output: Any, llm_output: Any, tolerance: float) -> bool:
    """
    Compare two outputs within a specified tolerance.

    Supports:
    - scalars
    - NumPy arrays
    - lists, tuples
    - nested structures like (array, array) vs object-array of arrays
    """
    try:
        norm_ref = _normalize_output(ref_output)
        norm_llm = _normalize_output(llm_output)
        return _compare_normalized(norm_ref, norm_llm, tolerance)
    except Exception:
        return False


def _normalize_output(obj: Any) -> Any:
    """Convert nested outputs into comparable canonical form."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return tuple(_normalize_output(x) for x in obj)
        return np.asarray(obj, dtype=float)
    if isinstance(obj, list):
        return [_normalize_output(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize_output(x) for x in obj)
    return obj


def _compare_normalized(a: Any, b: Any, tol: float) -> bool:
    """Compare normalized values recursively."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        return np.allclose(a, b, atol=tol, rtol=tol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_compare_normalized(x, y, tol) for x, y in zip(a, b))
    return a == b

@dataclass
class TaskEvaluationResult:
    """Complete result of task evaluation."""
    task_id: str
    verification_result: VerificationResult
    reference_code: str
    llm_code: str
    dependency_mode: str
    success: bool
    error_message: Optional[str] = None


def evaluate_task_implementation(
    task_yaml_path: Union[str, Path],
    llm_output: Union[str, ParsedCode],
    tasks_directory: Union[str, Path],
    environment: Environment,
    dependency_mode: DependencyMode = DependencyMode.REFERENCE_PROVIDED,
    llm_outputs: Optional[Dict[str, ParsedCode]] = None
) -> TaskEvaluationResult:
    """
    Complete evaluation pipeline for a single task implementation.
    
    Args:
        task_yaml_path: Path to the task YAML file
        llm_output: Raw LLM output string or ParsedCode object containing the implementation
        tasks_directory: Directory containing all task files (for dependencies)
        environment: Environment configuration
        dependency_mode: How to handle dependencies (REFERENCE_PROVIDED or LLM_CHAIN)
        llm_outputs: Dict mapping task_id -> ParsedCode (required for LLM_CHAIN mode)
        
    Returns:
        TaskEvaluationResult with verification results and metadata
        
    Examples:
        >>> env = load_environment("tier1_environment.yaml")
        >>> result = evaluate_task_implementation(
        ...     "T1_SF_001.yaml",
        ...     llm_output,
        ...     "tasks/",
        ...     env,
        ...     DependencyMode.REFERENCE_PROVIDED
        ... )
        >>> if result.success and result.verification_result.all_passed:
        ...     # All tests passed
        ...     pass
        ... else:
        ...     # Handle failure
        ...     pass
    """
    task_yaml_path = Path(task_yaml_path)
    tasks_directory = Path(tasks_directory)
    
    try:
        # Step 1: Load task from YAML
        task = load_task(task_yaml_path)
        
        # Step 2: Handle LLM output - if it's already a ParsedCode object, use it directly
        if isinstance(llm_output, ParsedCode):
            parsed_llm_output = llm_output
        elif isinstance(llm_output, str):
            # For string inputs, try to parse as JSON first
            try:
                parsed_llm_output = parse_llm_json_output(llm_output)
            except Exception:
                # If JSON parsing fails, this might be raw code - create a simple ParsedCode
                # This is a fallback for cases where we get raw function code
                parsed_llm_output = ParsedCode(
                    function_imports=[],
                    test_imports=[],
                    main_function=llm_output,
                    test_functions={},
                    main_function_name=task.expected_function_name,
                    all_imports=[]
                )
        else:
            raise ValueError(f"llm_output must be string or ParsedCode, got {type(llm_output)}")
        
        # Step 3: Construct reference code block
        reference_code = aggregate_reference_code(
            task=task,
            task_dir=tasks_directory,
            environment=environment
        )
        
        # Step 4: Construct LLM code block
        llm_code = aggregate_llm_code(
            task=task,
            task_dir=tasks_directory,
            environment=environment,
            llm_output=parsed_llm_output,
            dependency_mode=dependency_mode,
            llm_outputs=llm_outputs
        )
        
        # Step 5: Compare results according to test points
        verification_result = verify_against_reference(
            task=task,
            reference_code=reference_code,
            llm_code=llm_code
        )
        
        # Step 6: Return comprehensive results
        return TaskEvaluationResult(
            task_id=task.task_id,
            verification_result=verification_result,
            reference_code=reference_code,
            llm_code=llm_code,
            dependency_mode=dependency_mode,
            success=True,
            error_message=None
        )
        
    except Exception as e:
        # Handle any errors in the pipeline
        return TaskEvaluationResult(
            task_id=task_yaml_path.stem if task_yaml_path.exists() else "unknown",
            verification_result=VerificationResult(
                all_passed=False,
                test_results=[],
                total_tests=0,
                passed_tests=0
            ),
            reference_code="",
            llm_code="",
            dependency_mode=dependency_mode,
            success=False,
            error_message=str(e)
        )


@dataclass
class TestResult:
    """Result of running a single test function."""
    name: str
    passed: bool
    error: Optional[str] = None


@dataclass
class ReferenceTestResult:
    """Result of testing LLM tests against reference implementation."""
    all_passed: bool
    test_results: Dict[str, TestResult]


class TestEvaluator:
    """Fast in-memory test execution with pre-loaded imports."""
    
    def __init__(self, environment: Environment):
        """Initialize with environment imports pre-loaded."""
        self.base_namespace = {'__builtins__': __builtins__}
        
        # Pre-load all environment imports ONCE for speed
        for import_stmt in build_environment_imports(environment):
            exec(import_stmt, self.base_namespace)
    
    def run_tests_against_implementation(
        self,
        test_functions: Dict[str, str],
        implementation_code: str
    ) -> Dict[str, TestResult]:
        """
        Run test functions against specific implementation code.
        
        Args:
            test_functions: Dict mapping test_name -> test_code
            implementation_code: Implementation to test against
            
        Returns:
            Dict mapping test_name -> TestResult
        """
        # Copy base namespace (with imports already loaded)
        namespace = self.base_namespace.copy()
        
        try:
            # Execute implementation code
            exec(implementation_code, namespace)
        except Exception as e:
            # If implementation has syntax/import errors, all tests fail
            return {
                name: TestResult(name, passed=False, error=f"Implementation error: {e}")
                for name in test_functions.keys()
            }
        
        # Run each test function
        test_results = {}
        for test_name, test_code in test_functions.items():
            try:
                # Execute test function definition
                exec(test_code, namespace)
                
                # Get and call the test function
                test_func = namespace[test_name]
                test_func()  # If no exception, test passed
                
                test_results[test_name] = TestResult(test_name, passed=True)
                
            except Exception as e:
                test_results[test_name] = TestResult(
                    test_name, 
                    passed=False, 
                    error=str(e)
                )
        
        return test_results


@dataclass
class FailureDetectionResult:
    """Result of testing LLM tests against failure examples."""
    expected_failures_caught: List[ExpectedFailure]
    expected_failures_missed: List[ExpectedFailure]
    all_expected_caught: bool
    failure_test_results: Dict[str, Dict[str, TestResult]]  # failure_type -> test_name -> result


class TestQualityEvaluator:
    """Evaluator for testing LLM-generated tests against reference and failure examples."""
    
    def __init__(self, environment: Environment):
        """Initialize with pre-loaded environment imports for efficiency."""
        self.evaluator = TestEvaluator(environment)
    
    def test_against_reference(self, task: Task, parsed_llm_output: ParsedCode) -> ReferenceTestResult:
        """
        Test LLM-generated tests against reference implementation.
        All tests should pass.
        
        Args:
            task: Task object with reference solution
            parsed_llm_output: ParsedCode object with test functions
            
        Returns:
            ReferenceTestResult with pass/fail status
        """
        # Run LLM tests against reference implementation
        test_results = self.evaluator.run_tests_against_implementation(
            parsed_llm_output.test_functions,
            task.reference_solution.code
        )
        
        # Check if all tests passed
        all_passed = all(result.passed for result in test_results.values())
        
        return ReferenceTestResult(
            all_passed=all_passed,
            test_results=test_results
        )
    
    def run_tests_against_failures(self, task: Task, parsed_llm_output: ParsedCode) -> FailureDetectionResult:
        """
        Test LLM-generated tests against failure examples.
        Tests should fail on buggy implementations as specified in expected_failures.
        
        Args:
            task: Task object with failure examples and expected failures
            parsed_llm_output: ParsedCode object with test functions
            
        Returns:
            FailureDetectionResult with detection status
        """
        # Run LLM tests against each failure example
        failure_test_results = {}
        for failure_name, failure_code in task.failure_examples.items():
            failure_test_results[failure_name] = self.evaluator.run_tests_against_implementation(
                parsed_llm_output.test_functions,
                failure_code
            )
        
        # Check which expected failures were caught
        expected_failures_caught = []
        expected_failures_missed = []
        
        for expected in task.expected_failures:
            failure_type = expected.failure_type
            test_function = expected.test_function
            
            # Check if this expected failure was caught
            caught = False
            if failure_type in failure_test_results:
                test_result = failure_test_results[failure_type].get(test_function)
                if test_result and not test_result.passed:
                    caught = True
            
            if caught:
                expected_failures_caught.append(expected)
            else:
                expected_failures_missed.append(expected)
        
        # Check if all expected failures were caught
        all_expected_caught = len(expected_failures_missed) == 0
        
        return FailureDetectionResult(
            expected_failures_caught=expected_failures_caught,
            expected_failures_missed=expected_failures_missed,
            all_expected_caught=all_expected_caught,
            failure_test_results=failure_test_results
        )


def evaluate_single_task(
    task: Task,
    environment: Environment, 
    llm_output: ParsedCode,
    task_dir: Path,
    llm_outputs: Dict[str, ParsedCode] = None
) -> Dict[str, Any]:
    """
    Evaluate a single task and return summary dictionary.
    
    Args:
        task: Single task to evaluate
        environment: Environment configuration
        llm_output: LLM-generated code for this task
        task_dir: Directory containing all task YAML files (for dependencies)
        llm_outputs: Dict mapping task_id -> ParsedCode (for llm_chain mode)
        
    Returns:
        Dictionary with evaluation metrics for JSON export
    """
    # Create evaluator for test quality evaluation
    evaluator = TestQualityEvaluator(environment)
    
    # Test LLM tests against reference implementation
    ref_result = evaluator.test_against_reference(task, llm_output)
    
    # Test LLM tests against failure examples
    failure_result = evaluator.run_tests_against_failures(task, llm_output)
    
    # Calculate test quality metrics
    total_tests = len(llm_output.test_functions)
    tests_passed_on_reference = sum(1 for r in ref_result.test_results.values() if r.passed)
    total_expected_failures = len(task.expected_failures)
    expected_failures_failed_on_reference = len(failure_result.expected_failures_caught)
    
    # Evaluate LLM implementation correctness in both dependency modes
    
    # Mode 1: Reference provided (dependencies use reference implementations)
    try:
        ref_code = aggregate_reference_code(task, task_dir, environment)
        llm_code_ref_deps = aggregate_llm_code(
            task, task_dir, environment, llm_output, 
            DependencyMode.REFERENCE_PROVIDED
        )
        ref_provided_result = verify_against_reference(task, ref_code, llm_code_ref_deps)
        fcn_correct_with_reference_provided = 1 if ref_provided_result.all_passed else 0
    except Exception as e:
        # If dependency aggregation fails, mark as failed
        fcn_correct_with_reference_provided = 0
    
    # Mode 2: LLM chain (dependencies use LLM implementations)
    try:
        if llm_outputs is not None:
            llm_code_llm_deps = aggregate_llm_code(
                task, task_dir, environment, llm_output,
                DependencyMode.LLM_CHAIN, llm_outputs
            )
            llm_chain_result = verify_against_reference(task, ref_code, llm_code_llm_deps)
            fcn_correct_with_llm_chain = 1 if llm_chain_result.all_passed else 0
        else:
            # No LLM outputs provided, can't evaluate LLM chain
            fcn_correct_with_llm_chain = 0
    except Exception as e:
        # If LLM chain fails (missing dependencies, etc.), mark as failed
        fcn_correct_with_llm_chain = 0
    
    return {
        "task_id": task.task_id,
        "fcn_correct_with_reference_provided": fcn_correct_with_reference_provided,
        "fcn_correct_with_llm_chain": fcn_correct_with_llm_chain,
        "total_tests": total_tests,
        "tests_passed_on_reference": tests_passed_on_reference,
        "total_expected_failures": total_expected_failures,
        "expected_failures_failed_on_reference": expected_failures_failed_on_reference
    }
