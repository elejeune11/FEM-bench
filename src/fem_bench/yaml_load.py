# env_loader.py
# requirements:  pip install pyyaml "pydantic>=2"
import ast
import builtins
import numpy as np
from pathlib import Path
import re
from typing import List, Optional, Any
from typing import (
    get_origin,
    get_args,
    Dict,
    Tuple,
    Union,
    Callable as TypingCallable,
)
from collections.abc import Callable as AbcCallable
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
import inspect
import textwrap


# ---------------------------------------------------------------------------#
# 1.  Nested models with strict (= forbid extra keys)
# ---------------------------------------------------------------------------#

class Library(BaseModel):
    name: str
    version: str
    purpose: str
    import_as: Optional[str] = None
    usage: Optional[str] = None

    model_config = {
        "extra": "forbid"
    }


class TestingConfig(BaseModel):
    framework: str
    required_imports: List[str]
    allowed_imports: List[str]
    naming_convention: str

    model_config = {
        "extra": "forbid"
    }


class CodeRequirements(BaseModel):
    max_function_length: int
    docstring_required: bool
    vectorization_preferred: bool

    model_config = {
        "extra": "forbid"
    }


class Environment(BaseModel):
    environment_name: str
    tier: int
    description: str
    language: str
    python_version: str
    required_libraries: List[Library]
    allowed_libraries: List[Library] = Field(default_factory=list)
    testing: Optional[TestingConfig] = None
    code_requirements: Optional[CodeRequirements] = None
    import_guidelines: str = ""

    model_config = {
        "extra": "forbid"
    }


# ---------------------------------------------------------------------------#
# 2.  Environment Loader function
# ---------------------------------------------------------------------------#

def load_environment(env_path: Path) -> Environment:
    """
    Load a YAML file and return a validated `Environment` object.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    pydantic.ValidationError
        If the YAML violates the schema (missing keys, wrong types, extras).
    """
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    raw: Dict[str, Any] = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    return Environment.model_validate(raw)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def resolve_type(type_str: str) -> Any:
    scope = {
        "__builtins__": vars(builtins),
        "numpy": np,
        "np": np,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
    }

    try:
        # Reject nested brackets manually (not supported)
        if type_str.count("[") > 1 or type_str.count("]") > 1:
            raise ValueError(f"Nested generic types are not supported: {type_str}")

        # Match single generic form
        match = re.fullmatch(r"([\w\.]+)\[([\w\.]+)\]", type_str)
        if match:
            base_type_str, inner_type_str = match.groups()
            base_type = eval(base_type_str, scope)
            inner_type = eval(inner_type_str, scope)

            if base_type is not np.ndarray:
                raise TypeError(f"{base_type_str} is not supported for indexing")

            return (base_type, inner_type)

        # Non-generic fallback
        result = eval(type_str, scope)
        if isinstance(result, type) or callable(result):
            return result

        raise TypeError(f"{type_str} is not a type or callable")

    except Exception:
        raise ValueError(f"Unsupported type string: {type_str}")


def check_python_syntax(code: str, label: str = "code block"):
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax in {label}: {e}")


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class Parameter(BaseModel):
    name: str
    type: str
    shape: Optional[str] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        resolve_type(v)
        return v

    model_config = {"extra": "forbid"}


class FunctionSignature(BaseModel):
    input_parameters: List[Parameter]
    return_parameters: List[Parameter]

    model_config = {"extra": "forbid"}


class TaskDependency(BaseModel):
    function_name: str
    source_task: str

    model_config = {"extra": "forbid"}


class CodeBlock(BaseModel):
    code: str

    @field_validator("code")
    @classmethod
    def validate_code(cls, v):
        check_python_syntax(v)
        return v

    model_config = {"extra": "forbid"}


class TestCase(BaseModel):
    input: Dict[str, Any]
    tolerance: float = 1e-12

    model_config = {"extra": "forbid"}


class ExpectedFailure(BaseModel):
    failure_type: str
    test_function: str

    model_config = {"extra": "forbid"}


class Task(BaseModel):
    # Metadata
    task_id: str
    category: str
    subcategory: str
    title: str
    short_description: str
    version: str
    created_date: str
    created_by: str

    # Description & function spec
    prompt_description: str
    expected_function_name: str
    include_tests: bool
    expected_test_functions: List[str]
    function_signature: FunctionSignature

    # Dependencies (optional)
    task_dependencies: Optional[Dict[str, List[TaskDependency]]] = Field(default_factory=dict)

    # Reference implementation & failure examples
    reference_solution: CodeBlock
    failure_examples: Dict[str, CodeBlock]

    # Evaluation
    reference_verification: Dict[str, List[TestCase]]
    test_efficacy_verification: Dict[str, List[ExpectedFailure]]

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_test_inputs_against_signature(self) -> "Task":
        expected_inputs = {param.name for param in self.function_signature.input_parameters}
        for test_case in self.reference_verification.get("test_cases", []):
            actual_inputs = set(test_case.input)
            unknown_keys = actual_inputs - expected_inputs
            if unknown_keys:
                raise ValueError(f"Test case input has unexpected parameters: {unknown_keys}")
        return self

    @model_validator(mode="after")
    def normalize_failure_example_names(self) -> "Task":
        expected_name = self.expected_function_name
        expected_args = [param.name for param in self.function_signature.input_parameters]
        normalized_failures = {}

        for key, example in self.failure_examples.items():
            try:
                tree = ast.parse(example.code)
            except SyntaxError as e:
                raise ValueError(f"Invalid Python syntax in failure_examples[{key}]") from e

            fn_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            if not fn_defs:
                raise ValueError(f"failure_examples[{key}] must contain a function definition.")

            fn = fn_defs[0]
            arg_names = [a.arg for a in fn.args.args]

            if arg_names != expected_args:
                raise ValueError(
                    f"failure_examples[{key}] has argument names {arg_names}, expected {expected_args}"
                )

            # Rename function to match expected_function_name
            fn.name = expected_name

            # Rewrite code from AST to string
            new_code = ast.unparse(tree).strip()
            normalized_failures[key] = type(example)(code=new_code)

        self.failure_examples = normalized_failures
        return self


def load_task(task_path: Path) -> Task:
    raw = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    return Task.model_validate(raw)


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


def get_type_string(t: Any) -> str:
    """
    Convert a Python/typing type into a readable, YAML-friendly string that our
    `resolve_type()` helper (and Pydantic validators) can understand.

    Examples
    --------
    >>> get_type_string(int)
    'int'
    >>> get_type_string(List[int])
    'List[int]'
    >>> get_type_string(Dict[str, float])
    'Dict[str, float]'
    >>> get_type_string(Callable[[float], float])
    'Callable[[float], float]'
    """
    # ── 1. Special cases ──────────────────────────────────────────────────
    if t is np.ndarray:
        return "numpy.ndarray"
    if isinstance(t, type):                       # int, float, bool, custom classes …
        return t.__name__

    # ── 2. typing / collections-abc generics ──────────────────────────────
    origin = get_origin(t)
    args   = get_args(t)

    # List[…]
    if origin in {list, List}:
        return f"List[{get_type_string(args[0])}]" if args else "List"

    # Dict[…, …]
    if origin in {dict, Dict}:
        if len(args) == 2:
            key_str   = get_type_string(args[0])
            value_str = get_type_string(args[1])
            return f"Dict[{key_str}, {value_str}]"
        return "Dict"

    # Tuple[…]
    if origin in {tuple, Tuple}:
        if args and args[-1] is ...:                         # Tuple[int, ...]
            return f"Tuple[{get_type_string(args[0])}, ...]"
        return f"Tuple[{', '.join(get_type_string(a) for a in args)}]"

    # Union[…]
    if origin is Union:
        return "Union[" + ", ".join(get_type_string(a) for a in args) + "]"

    # Callable[[…], Ret]
    if origin in {AbcCallable, TypingCallable}:
        if len(args) == 2 and isinstance(args[0], (list, tuple)):
            arg_list    = ", ".join(get_type_string(a) for a in args[0])
            return_type = get_type_string(args[1])
            return f"Callable[[{arg_list}], {return_type}]"
        return "Callable"

    # ── 3. Fallbacks ──────────────────────────────────────────────────────
    if hasattr(t, "__name__"):
        return t.__name__

    # Last-resort: strip the leading "typing." for readability
    return str(t).replace("typing.", "")


def extract_signature(fn: Callable) -> FunctionSignature:
    sig = inspect.signature(fn)
    input_parameters = []
    for name, param in sig.parameters.items():
        if param.annotation is inspect.Parameter.empty:
            raise ValueError(f"Parameter '{name}' in function '{fn.__name__}' is missing a type annotation.")
        type_str = get_type_string(param.annotation)
        input_parameters.append(Parameter(
            name=name,
            type=type_str,
            shape=None
        ))

    if sig.return_annotation is inspect.Signature.empty:
        raise ValueError(f"Function '{fn.__name__}' is missing a return type annotation.")

    return_type_str = get_type_string(sig.return_annotation)

    return_parameters = [
        Parameter(
            name="result",
            type=return_type_str,
            shape=None
        )
    ]

    return FunctionSignature(
        input_parameters=input_parameters,
        return_parameters=return_parameters
    )


def get_source(fn: Callable) -> str:
    return textwrap.dedent(inspect.getsource(fn))


def create_task_from_functions(
    task_id: str,
    reference_fn: Callable,
    failure_fns: Dict[str, Callable],
    *,
    title: str = "",
    short_description: str = "",
    created_by: str = "auto",
    created_date: str = "2025-06-23",
    **task_overrides
) -> Task:
    fn_name = reference_fn.__name__
    signature = extract_signature(reference_fn)

    # Generate default test input values (all 0.0)
    default_inputs = {
        param.name: 0.0 for param in signature.input_parameters
    }

    default_task_fields = {
        "task_id": task_id,
        "category": "generated",
        "subcategory": "auto",
        "title": title or f"Auto task for {fn_name}",
        "short_description": short_description or f"Generated task for {fn_name}",
        "version": "1.0",
        "created_date": created_date,
        "created_by": created_by,
        "prompt_description": "",
        "expected_function_name": fn_name,
        "include_tests": True,
        "expected_test_functions": [f"test_{fn_name}"],
        "function_signature": signature,
        "task_dependencies": {},
        "reference_solution": CodeBlock(code=get_source(reference_fn)),
        "failure_examples": {
            name: CodeBlock(code=get_source(fn))
            for name, fn in failure_fns.items()
        },
        "reference_verification": {
            "test_cases": [
                TestCase(input=default_inputs, tolerance=1e-12)
            ]
        },
        "test_efficacy_verification": {
            "expected_failures": [
                ExpectedFailure(failure_type=name, test_function=f"test_{fn_name}")
                for name in failure_fns
            ]
        }
    }

    # Let user override any fields
    default_task_fields.update(task_overrides)
    return Task(**default_task_fields)


def dump_task_to_yaml(task: Task, path: Path) -> None:
    """
    Dump a Task model to a YAML file at the given path.

    Parameters:
    - task: a validated Task Pydantic object
    - path: the output Path object (e.g., Path("T1_SF_001.yaml"))
    """
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            task.model_dump(mode="json"),
            f,
            sort_keys=False,
            indent=2
        )
