"""
Task and environment loader for FEM-Bench YAML files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numbers
import yaml
import numpy as np   # NEW

# --------------------------------------------------------------------------- #
#                               Data classes                                  #
# --------------------------------------------------------------------------- #

@dataclass
class Library:
    name: str
    version: str
    import_as: Optional[str]
    purpose: str
    usage: Optional[str] = None


@dataclass
class Environment:
    environment_name: str
    tier: int
    description: str
    language: str
    python_version: str
    required_libraries: List[Library]
    allowed_libraries: List[Library]
    testing: Dict[str, Any]
    code_requirements: Dict[str, Any]
    import_guidelines: str


@dataclass
class FunctionSignature:
    parameters: List[str]
    parameter_types: List[str]
    return_shape: str


@dataclass
class TaskDependency:
    source_task: str


@dataclass
class TestCase:
    input: Dict[str, Any]
    tolerance: float = 1e-12


@dataclass
class ExpectedFailure:
    failure_type: str
    test_function: str


@dataclass
class ReferenceSolution:
    code: str


@dataclass
class Task:
    # Metadata
    task_id: str
    category: str
    subcategory: str
    title: str
    short_description: str
    version: str
    created_date: str
    created_by: str

    # Task specification
    prompt: str
    expected_function_name: str
    include_tests: bool
    expected_test_functions: List[str]
    function_signature: FunctionSignature
    task_dependencies: List[TaskDependency]

    # Reference solutions
    reference_solution: ReferenceSolution
    failure_examples: Dict[str, str]

    # Evaluation configuration
    test_cases: List[TestCase]
    expected_failures: List[ExpectedFailure]


# --------------------------------------------------------------------------- #
#                           Helper: auto-NumPy                                #
# --------------------------------------------------------------------------- #
def _auto_numpy(val: Any) -> Any:
    """(unchanged) Convert plain lists → ndarray when safe."""
    if isinstance(val, list):
        if all(isinstance(x, numbers.Number) for x in val):
            return np.asarray(val, dtype=float)
        return [_auto_numpy(v) for v in val]
    if isinstance(val, dict):
        return {k: _auto_numpy(v) for k, v in val.items()}
    return val


def _coerce_by_signature(value: Any, type_hint: str | None) -> Any:
    """
    Convert *value* to the Python object indicated by *type_hint*
    (“np.ndarray”, “float”, “int”, …). Falls back to _auto_numpy.
    """
    if not type_hint:                       # nothing declared → best-effort
        return _auto_numpy(value)

    norm = type_hint.replace("numpy.", "np.").strip().lower()

    # --- ndarray -----------------------------------------------------------
    if norm.startswith("np.ndarray"):
        arr = np.asarray(_auto_numpy(value))
        # Try to preserve int dtype when obvious, else float
        if arr.dtype.kind in ("i", "u"):
            return arr.astype(int)
        return arr.astype(float)

    # --- scalar numerics ---------------------------------------------------
    if norm in {"float", "np.float64", "np.float32"}:
        return float(value)
    if norm in {"int", "np.int64", "np.int32"}:
        return int(value)

    # --- fallback ----------------------------------------------------------
    return _auto_numpy(value)


# --------------------------------------------------------------------------- #
#                              Task loader                                    #
# --------------------------------------------------------------------------- #
def load_task(task_path: Path) -> Task:
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")

    with task_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    # --- Required fields --------------------------------------------------
    required = [
        "task_id", "category", "subcategory", "title", "short_description",
        "prompt", "expected_function_name", "function_signature",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # --- Function signature ----------------------------------------------
    sig = data["function_signature"]
    func_sig = FunctionSignature(
        parameters=sig.get("parameters", []),
        parameter_types=sig.get("parameter_types", []),
        return_shape=sig.get("return_shape", ""),
    )

    # --- Dependencies -----------------------------------------------------
    deps_yaml = data.get("task_dependencies", {}).get("required_functions", [])
    deps = [TaskDependency(source_task=d["source_task"]) for d in deps_yaml]

    # --- Test cases -------------------------------------------------------
    param_type_map = dict(zip(func_sig.parameters, func_sig.parameter_types))

    tc_yaml = data.get("reference_verification", {}).get("test_cases", [])
    test_cases = []
    for case in tc_yaml:
        coerced_input = {
            name: _coerce_by_signature(val, param_type_map.get(name))
            for name, val in case["input"].items()
        }
        test_cases.append(
            TestCase(
                input=coerced_input,
                tolerance=float(case.get("tolerance", 1e-12)),
            )
        )

    # --- Expected failures ------------------------------------------------
    fail_yaml = data.get("test_efficacy_verification", {}).get("expected_failures", [])
    exp_fail = [
        ExpectedFailure(failure_type=f["failure_type"], test_function=f["test_function"])
        for f in fail_yaml
    ]

    # --- Reference solution ----------------------------------------------
    ref_sol_yaml = data.get("reference_solution", {})
    ref_solution = ReferenceSolution(code=ref_sol_yaml.get("code", ""))

    # --- Build and return -------------------------------------------------
    return Task(
        task_id=data["task_id"],
        category=data["category"],
        subcategory=data["subcategory"],
        title=data["title"],
        short_description=data["short_description"],
        version=data.get("version", "1.0"),
        created_date=data.get("created_date", ""),
        created_by=data.get("created_by", ""),
        prompt=data["prompt"],
        expected_function_name=data["expected_function_name"],
        include_tests=data.get("include_tests", True),
        expected_test_functions=data.get("expected_test_functions", []),
        function_signature=func_sig,
        task_dependencies=deps,
        reference_solution=ref_solution,
        failure_examples=data.get("failure_examples", {}),
        test_cases=test_cases,
        expected_failures=exp_fail,
    )


# --------------------------------------------------------------------------- #
#                          Bulk task loader                                   #
# --------------------------------------------------------------------------- #
def load_all_tasks(tasks_dir: Path) -> List[Task]:
    tasks: List[Task] = []
    for yaml_file in tasks_dir.glob("*.yaml"):
        if yaml_file.name.startswith("_"):   # Skip template files
            continue
        try:
            tasks.append(load_task(yaml_file))
        except Exception as exc:
            print(f"Warning: Failed to load {yaml_file}: {exc}")
    return sorted(tasks, key=lambda t: t.task_id)


# --------------------------------------------------------------------------- #
#                      Environment loader (unchanged)                         #
# --------------------------------------------------------------------------- #
def load_environment(env_path: Path) -> Environment:
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    with env_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    required = [
        "environment_name", "tier", "description", "language",
        "python_version", "required_libraries",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    def _parse_libs(lst: List[Dict]) -> List[Library]:
        return [
            Library(
                name=lib["name"],
                version=lib["version"],
                import_as=lib.get("import_as"),
                purpose=lib["purpose"],
                usage=lib.get("usage"),
            )
            for lib in lst
        ]

    required_libs = _parse_libs(data["required_libraries"])
    allowed_libs  = _parse_libs(data.get("allowed_libraries", []))

    return Environment(
        environment_name=data["environment_name"],
        tier=data["tier"],
        description=data["description"],
        language=data["language"],
        python_version=data["python_version"],
        required_libraries=required_libs,
        allowed_libraries=allowed_libs,
        testing=data.get("testing", {}),
        code_requirements=data.get("code_requirements", {}),
        import_guidelines=data.get("import_guidelines", ""),
    )
