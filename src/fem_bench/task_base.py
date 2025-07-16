import ast
from typing import Any, Dict, List, Optional


class CodeBlock:
    def __init__(self, main_function: str, preamble: Optional[str] = None):
        self.preamble = preamble or ""
        self.main_function = main_function
        self.full_code = self.preamble + "\n" + self.main_function

        self.tree = ast.parse(self.main_function)
        self.func_node = self._get_function_node()

        if self.func_node is None:
            raise ValueError("No function found in the provided main_function block.")

    def _get_function_node(self) -> Optional[ast.FunctionDef]:
        """Assumes exactly one function per main block."""
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                return node
        return None

    def get_signature(self) -> Dict[str, Any]:
        args = [
            {
                "name": arg.arg,
                "type": ast.unparse(arg.annotation) if arg.annotation else None
            }
            for arg in self.func_node.args.args
        ]
        return_type = ast.unparse(self.func_node.returns) if self.func_node.returns else None

        return {
            "name": self.func_node.name,
            "args": args,
            "returns": return_type
        }

    def get_docstring(self) -> str:
        return ast.get_docstring(self.func_node) or ""

    def execute(self) -> Dict[str, Any]:
        """Executes preamble + function in one namespace."""
        namespace = {}
        exec(self.full_code, namespace)
        return namespace


class Task:
    def __init__(
        self,
        task_id: str,
        task_short_description: str,
        created_date: str,
        created_by: str,
        main_fcn_code: str,
        required_imports: Optional[List[str]] = None,
        fcn_dependency_code: Optional[List[str]] = None,
        reference_verification_inputs: Optional[List[List]] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ):
        # --- Metadata ---
        self.task_id = task_id
        self.task_short_description = task_short_description
        self.created_date = created_date
        self.created_by = created_by

        # --- Execution Spec (source only) ---
        self.main_fcn_code = main_fcn_code.strip()
        self.required_imports = required_imports or []
        self.fcn_dependency_code = fcn_dependency_code or []

        # --- Verification ---
        self.reference_verification_inputs = reference_verification_inputs or []

        # Each test case is a dict with:
        # - "test_code": str (required)
        # - "expected_failures": List[str] (required, may be empty)
        self.test_cases = test_cases or []

    def __repr__(self):
        return f"<Task {self.task_id}: {self.task_short_description}>"
