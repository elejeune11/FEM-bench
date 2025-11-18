import ast
from collections import defaultdict
from fem_bench.task_loader import load_task_from_info
from fem_bench.task_to_prompt import task_to_code_prompt, task_to_test_prompt
from fem_bench.evaluate_output import evaluate_function_output_match, evaluate_task_tests
from fem_bench.evaluate_output import load_function_from_code, run_test_case
from importlib.util import spec_from_file_location, module_from_spec
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):  # handles np.float64, np.int32, etc.
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def validate_syntax(code: str) -> tuple[bool, str | None]:
    """
    Check whether a string of Python code is syntactically valid.

    Returns:
        (bool, error): True if valid, False otherwise. `error` contains the SyntaxError message if invalid.
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


class FEMBenchPipeline:
    def __init__(
        self,
        tasks_dir: str,
        llm_outputs_dir: str,
        prompts_dir: str,
        results_dir: str,
        template_dir: str,
        template_name: str
    ):
        # --- Directory setup ---
        self.tasks_dir = Path(tasks_dir)
        self.llm_outputs_dir = Path(llm_outputs_dir)
        self.prompts_dir = Path(prompts_dir)
        self.results_dir = Path(results_dir)
        self.template_dir = Path(template_dir)
        self.template_name = template_name

        # Create directories if they don’t exist
        for d in [self.prompts_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # --- Internal state ---
        self.tasks = {}         # task_id -> Task object
        self.prompts = {}       # task_id -> {"code": ..., "tests": ...}
        self.llm_outputs = {}   # task_id -> generated function code (str)
        self.results = {}       # task_id -> eval results (dict)

    def __repr__(self):
        return (
            f"FEMBenchPipeline("
            f"tasks={len(self.tasks)}, "
            f"llm_outputs={len(self.llm_outputs)}, "
            f"results={len(self.results)})"
        )

    def load_all_tasks(self):
        """
        Load all task modules from self.tasks_dir and populate self.tasks with Task objects.
        Each task module must define a `task_info()` function.
        """
        for py_file in self.tasks_dir.glob("*.py"):
            module_name = py_file.stem
            spec = spec_from_file_location(module_name, py_file)
            module = module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, "task_info"):
                raise ValueError(f"Module {py_file.name} is missing a `task_info()` function.")

            task = load_task_from_info(module.task_info)
            self.tasks[task.task_id] = task

    def generate_and_save_task_prompts(self):
        """
        Generate LLM code-generation prompts for each task and save them to disk.
        Saved files are written to {self.prompts_dir}/{task_id}_code_prompt.txt.
        """
        for task_id, task in self.tasks.items():
            prompt = task_to_code_prompt(task, self.template_dir, self.template_name)
            prompt_path = self.prompts_dir / f"{task_id}_code_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            self.prompts.setdefault(task_id, {})["code"] = prompt

    def generate_and_save_test_prompts(self):
        """
        Generate LLM test-generation prompts for each task and save them to disk.
        Saved files are written to {self.prompts_dir}/{task_id}_test_prompt.txt.
        """
        for task_id, task in self.tasks.items():
            prompt = task_to_test_prompt(task)
            prompt_path = self.prompts_dir / f"{task_id}_test_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            self.prompts.setdefault(task_id, {})["tests"] = prompt

    @staticmethod
    def extract_function_code(source: str) -> str:
        """Strip imports and extract the first function definition from a code string."""
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return ast.unparse(node)
        return None

    @staticmethod
    def extract_test_functions(source: str) -> Dict[str, str]:
        """Return a dict mapping test function names to their code blocks."""
        tree = ast.parse(source)
        func_map = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                func_map[node.name] = ast.unparse(node)
        return func_map

    def load_all_llm_outputs(self, allowed_llms: list[str] = None):

        for file in self.llm_outputs_dir.glob("*.py"):
            stem = file.stem
            content = file.read_text(encoding="utf-8")

            if "_code_" in stem:
                task_id, llm_name = stem.split("_code_", 1)
                key = "code"

                valid, err = validate_syntax(content)
                if not valid:
                    print(f"[SyntaxError] Skipping code file {file.name}: {err}")
                    self.results.setdefault(task_id, {})[llm_name] = {
                        "task_id": task_id,
                        "llm_name": llm_name,
                        "matches_reference": False,
                        "error": f"SyntaxError in function code: {err}",
                        "test_results": []
                    }
                    continue

                code_str = self.extract_function_code(content)
                if code_str is None:
                    reason = "No function definition found in the code."
                    first_line = content.strip().splitlines()[0] if content.strip() else ""
                    if first_line.startswith("#"):
                        reason = f"{reason} Marker: {first_line[1:].strip()}"
                    print(f"[Info] Skipping code file {file.name}: {reason}")
                    self.results.setdefault(task_id, {})[llm_name] = {
                        "task_id": task_id,
                        "llm_name": llm_name,
                        "matches_reference": False,
                        "error": reason,
                        "test_results": []
                    }
                    continue

            elif "_test_" in stem:
                task_id, llm_name = stem.split("_test_", 1)
                key = "test"

                valid, err = validate_syntax(content)
                if not valid:
                    print(f"[SyntaxError] Skipping test file {file.name}: {err}")
                    self.results.setdefault(task_id, {})[llm_name] = {
                        "task_id": task_id,
                        "llm_name": llm_name,
                        "tests": {
                            "tests_run": False,
                            "error": f"SyntaxError in test code: {err}",
                            "test_results": {}
                        }
                    }
                    continue

                code_str = self.extract_test_functions(content)
                if not code_str:  # empty dict
                    reason = "No test functions found in the file."
                    first_line = content.strip().splitlines()[0] if content.strip() else ""
                    if first_line.startswith("#"):
                        reason = f"{reason} Marker: {first_line[1:].strip()}"
                    print(f"[Info] Skipping test file {file.name}: {reason}")
                    self.results.setdefault(task_id, {})[llm_name] = {
                        "task_id": task_id,
                        "llm_name": llm_name,
                        "tests": {
                            "tests_run": False,
                            "error": reason,
                            "test_results": {}
                        }
                    }
                    continue

            else:
                continue

            if allowed_llms is not None and llm_name not in allowed_llms:
                continue

            self.llm_outputs.setdefault(task_id, {}).setdefault(llm_name, {})[key] = code_str

    def evaluate_all_llm_outputs(self):
        """
        Evaluate all LLM-generated functions against reference implementations.

        Stores results as a nested dict: results[task_id][llm_name] = { ... }
        Also writes each result to a file: results_dir/{task_id}_eval_{llm_name}.json
        """
        for task_id, llm_dict in self.llm_outputs.items():
            task = self.tasks.get(task_id)
            if not task:
                print(f"[Warning] Skipping unknown task_id: {task_id}")
                continue

            try:
                ref_fn = load_function_from_code(
                    task.main_fcn_code,
                    required_imports=task.required_imports,
                    fcn_dependencies=task.fcn_dependency_code
                )
            except Exception as e:
                print(f"[Error] Failed to load reference function for {task_id}: {e}")
                continue

            for llm_name, blocks in llm_dict.items():
                result = {
                    "task_id": task_id,
                    "llm_name": llm_name,
                    "matches_reference": False,
                    "test_results": []
                }

                try:
                    gen_fn = load_function_from_code(
                        blocks["code"],
                        required_imports=task.required_imports,
                        fcn_dependencies=task.fcn_dependency_code
                    )
                    
                    ok, detailed_results = evaluate_function_output_match(
                        ref_fn,
                        gen_fn,
                        task.reference_verification_inputs
                    )
                    
                    result["matches_reference"] = ok
                    result["test_results"] = detailed_results

                except Exception as e:
                    result["error"] = str(e)

                # Save in nested results dict
                self.results.setdefault(task_id, {})[llm_name] = result

                # Write to file
                out_path = self.results_dir / f"{task_id}_eval_{llm_name}.json"
                out_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    def evaluate_all_llm_tests(self):
        """
        Evaluate all LLM‑generated test files for each task.

        For every (task, LLM) pair:
        1. Load the reference implementation.
        2. Merge the LLM‑generated tests with any *expected‑failure* snippets that
           belonged to the same original test name.
        3. Run the tests with `evaluate_task_tests`.
        4. Persist results in memory and on disk.

        A row is added to ``failure_fail`` only when a test’s
        ``expected_failures`` list is *non‑empty*, so preserving that mapping is
        essential for correct scoring.
        """

        def _first_func_name(src: str) -> str:
            """Return the first function name defined in *src*, or '' on error."""
            try:
                tree = ast.parse(src)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        return node.name
            except Exception:
                pass
            return ""

        for task_id, llm_dict in self.llm_outputs.items():
            task = self.tasks.get(task_id)
            if not task:
                print(f"[Warning] Skipping unknown task_id: {task_id}")
                continue

            # ----------------------------------
            # 1. load the reference function
            # ----------------------------------
            try:
                reference_fcn = load_function_from_code(
                    task.main_fcn_code,
                    required_imports=task.required_imports,
                    fcn_dependencies=task.fcn_dependency_code
                )
            except Exception as e:
                print(f"[Error] Failed to load reference function for {task_id}: {e}")
                continue

            # Build a map {original_test_name: [expected_failure_codes]}
            original_fail_map = {
                _first_func_name(case["test_code"]): case.get("expected_failures", [])
                for case in task.test_cases
            }

            for llm_name, blocks in llm_dict.items():
                result = {
                    "task_id": task_id,
                    "llm_name": llm_name,
                    "tests_run": True,
                    "test_results": {},
                }

                try:
                    # -----------------------------
                    # 2. merge tests + failures
                    # -----------------------------
                    test_dict = blocks.get("test", {})
                    merged_cases = []
                    for test_src in test_dict.values():
                        name = _first_func_name(test_src)
                        exp_fail = original_fail_map.get(name, [])

                        # Warn once if the LLM invented a new test name
                        if name and name not in original_fail_map and exp_fail == []:
                            print(
                                f"[Info] LLM '{llm_name}' produced a new test "
                                f"'{name}' for task '{task_id}' (no expected failures)."
                            )

                        merged_cases.append(
                            {
                                "test_code": test_src,
                                "expected_failures": exp_fail,
                            }
                        )

                    # Install the merged list on the Task object
                    task.test_cases = merged_cases

                    # -----------------------------
                    # 3. run the evaluation
                    # -----------------------------
                    test_eval = evaluate_task_tests(task, reference_fcn)
                    result["test_results"] = test_eval

                except Exception as e:
                    result["tests_run"] = False
                    result["error"] = str(e)

                # -----------------------------
                # 4. store + persist
                # -----------------------------
                self.results.setdefault(task_id, {}).setdefault(llm_name, {})[
                    "tests"
                ] = result

                out_path = self.results_dir / f"{task_id}_tests_{llm_name}.json"
                out_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    def compute_aggregate_score(self) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregate scores from evaluation results.

        Each LLM gets:
            - fcn_correct_pct                    : % of (task, llm) pairs with correct function implementations
            - avg_tests_passed_on_reference_pct  : average % of reference tests passed
            - avg_expected_failures_detected_pct : average % of expected failures correctly failed
            - avg_joint_success_pct              : average % of tests that passed ref *and* failed all expected failures

        The score dict for each LLM is also written to
            results_dir/llm_aggregate_{llm_name}.json
        """
        # --- Collect universe of tasks and LLM names ---------------------------------
        all_task_ids = list(self.results.keys())
        all_llm_names = {llm for task in self.results.values() for llm in task.keys()}

        # --- Accumulators -------------------------------------------------------------
        counters = defaultdict(lambda: {
            "fcn_total": 0,
            "fcn_correct": 0,
            "test_pass_rates": [],          # list of floats (0–1) – one per task
            "failure_detection_rates": [],  # list of floats (0–1) – one per task
            "joint_success_rates": [],      # list of floats (0–1) – one per task
        })

        # --- Score every (task, llm) pair --------------------------------------------
        for task_id in all_task_ids:
            task_result = self.results[task_id]
            for llm_name in all_llm_names:
                c = counters[llm_name]
                c["fcn_total"] += 1

                eval_blocks = task_result.get(llm_name)

                # 1) Function correctness
                if eval_blocks and eval_blocks.get("matches_reference", False):
                    c["fcn_correct"] += 1

                # 2) Tests & failure detection
                test_results = (eval_blocks or {}).get("tests", {}).get("test_results", {})
                ref_pairs = test_results.get("reference_pass", [])
                fail_pairs = test_results.get("failure_fail", [])

                # Reference-pass rate
                if ref_pairs:
                    n_passed = sum(ok for _, ok in ref_pairs)
                    c["test_pass_rates"].append(n_passed / len(ref_pairs))
                else:
                    c["test_pass_rates"].append(0.0)

                # Failure-detection rate
                if fail_pairs:
                    n_failed = sum(ok for _, ok in fail_pairs)
                    c["failure_detection_rates"].append(n_failed / len(fail_pairs))
                else:
                    c["failure_detection_rates"].append(0.0)

                # 3) Joint success computation
                ref_dict = dict(ref_pairs)
                fail_dict = dict(fail_pairs)

                joint_total = 0
                joint_pass = 0

                for test_name in ref_dict:
                    if test_name in fail_dict:
                        joint_total += 1
                        if ref_dict[test_name] and fail_dict[test_name]:
                            joint_pass += 1

                if joint_total > 0:
                    c["joint_success_rates"].append(joint_pass / joint_total)
                else:
                    c["joint_success_rates"].append(0.0)

        # --- Finalise per-LLM metrics -------------------------------------------------
        llm_metrics = {}
        for llm_name, c in counters.items():
            fcn_pct = 100.0 * c["fcn_correct"] / c["fcn_total"]
            test_pct = 100.0 * sum(c["test_pass_rates"]) / len(c["test_pass_rates"])
            ef_pct = 100.0 * sum(c["failure_detection_rates"]) / len(c["failure_detection_rates"])
            joint_pct = 100.0 * sum(c["joint_success_rates"]) / len(c["joint_success_rates"])

            llm_metrics[llm_name] = {
                "fcn_correct_pct": round(fcn_pct, 2),
                "avg_tests_passed_on_reference_pct": round(test_pct, 2),
                "avg_expected_failures_detected_pct": round(ef_pct, 2),
                "avg_joint_success_pct": round(joint_pct, 2),
            }

            out_file = self.results_dir / f"llm_aggregate_{llm_name}.json"
            out_file.write_text(json.dumps(llm_metrics[llm_name], indent=2), encoding="utf-8")

        return llm_metrics

    def create_markdown_summary(
        self,
        filename: str = "evaluation_summary.md",
        model_names: Optional[list[str]] = None,
    ) -> None:
        """
        Write a Markdown summary with two tables:
        1. Function correctness (✓/×)
        2. Joint Test Success Rate (%): test passes on reference AND fails all expected failures

        If model_names is provided (e.g., ["gpt-4o","gemini-2.5-pro","claude-3-5","deepseek-chat"]),
        only those models (in that order) are included in the tables. Models with no results for a task
        show × for correctness and – for joint (counted as 0% in the average).
        """
        from collections import defaultdict
        import pandas as pd

        task_ids = sorted(self.results.keys())

        if model_names is not None:
            # Use exactly the given list (keep order, allow missing data)
            llm_names = list(model_names)
        else:
            # Fallback: infer from results (sorted)
            llm_names = sorted({llm for task in self.results.values() for llm in task})

        # Build row data and numeric buckets
        code_rows, joint_rows = [], []
        joint_numeric = defaultdict(list)  # llm -> list[float]

        for tid in task_ids:
            r_code, r_joint = [tid], [tid]

            for llm in llm_names:
                res = self.results.get(tid, {}).get(llm, {})
                tests = res.get("tests", {}).get("test_results", {})

                # Function correctness
                r_code.append("✓" if res.get("matches_reference") else "×")

                # Joint success
                ref_dict = dict(tests.get("reference_pass", []))
                fail_dict = dict(tests.get("failure_fail", []))

                joint_total = 0
                joint_pass = 0
                for test_name in ref_dict:
                    if test_name in fail_dict:
                        joint_total += 1
                        if ref_dict[test_name] and fail_dict[test_name]:
                            joint_pass += 1

                if joint_total > 0:
                    pct = 100.0 * joint_pass / joint_total
                    r_joint.append(f"{pct:.1f}%")
                    joint_numeric[llm].append(pct)
                else:
                    r_joint.append("–")
                    joint_numeric[llm].append(0.0)  # Always count as 0% in the average

            code_rows.append(r_code)
            joint_rows.append(r_joint)

        # Aggregate rows
        total_code = ["Total"]
        total_joint = ["Avg Joint Success %"]

        for llm in llm_names:
            total_tasks = len(task_ids)
            correct_tasks = sum(
                1
                for tid in task_ids
                if self.results.get(tid, {}).get(llm, {}).get("matches_reference")
            )
            total_code.append(f"{correct_tasks}/{total_tasks}")

            joint_vals = joint_numeric[llm] or [0.0] * total_tasks
            total_joint.append(f"{sum(joint_vals)/len(joint_vals):.1f}%")

        code_rows.append(total_code)
        joint_rows.append(total_joint)

        # Convert to Markdown
        def md_table(rows, hdr, title):
            return f"### {title}\n\n" + pd.DataFrame(rows, columns=hdr).to_markdown(index=False) + "\n\n"

        headers = ["Task"] + llm_names
        md = (
            md_table(code_rows, headers, "Function Correctness (✓ = Match)")
            + md_table(joint_rows, headers, "Joint Test Success Rate (%)")
        )

        (self.results_dir / filename).write_text(md, encoding="utf-8")