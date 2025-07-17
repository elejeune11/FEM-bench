import ast
from fem_bench.task_loader import load_task_from_info
from fem_bench.task_to_prompt import task_to_code_prompt, task_to_test_prompt
from fem_bench.evaluate_output import evaluate_function_output_match, evaluate_task_tests
from fem_bench.evaluate_output import load_function_from_code, run_test_case
from importlib.util import spec_from_file_location, module_from_spec
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


class FEMBenchPipeline:
    def __init__(
        self,
        tasks_dir: str,
        llm_outputs_dir: str,
        prompts_dir: str,
        results_dir: str,
    ):
        # --- Directory setup ---
        self.tasks_dir = Path(tasks_dir)
        self.llm_outputs_dir = Path(llm_outputs_dir)
        self.prompts_dir = Path(prompts_dir)
        self.results_dir = Path(results_dir)

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
            prompt = task_to_code_prompt(task)
            prompt_path = self.prompts_dir / f"{task_id}_code_prompt.txt"
            prompt_path.write_text(prompt)
            self.prompts.setdefault(task_id, {})["code"] = prompt

    def generate_and_save_test_prompts(self):
        """
        Generate LLM test-generation prompts for each task and save them to disk.
        Saved files are written to {self.prompts_dir}/{task_id}_test_prompt.txt.
        """
        for task_id, task in self.tasks.items():
            prompt = task_to_test_prompt(task)
            prompt_path = self.prompts_dir / f"{task_id}_test_prompt.txt"
            prompt_path.write_text(prompt)
            self.prompts.setdefault(task_id, {})["tests"] = prompt

    @staticmethod
    def extract_function_code(source: str) -> str:
        """Strip imports and extract the first function definition from a code string."""
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return ast.unparse(node)
        raise ValueError("No function definition found in the code.")

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
                code_str = self.extract_function_code(content)

            elif "_test_" in stem:
                task_id, llm_name = stem.split("_test_", 1)
                key = "test"
                code_str = self.extract_test_functions(content)

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
                out_path.write_text(json.dumps(result, indent=2))

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
                    required_imports=task.required_imports
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
                out_path.write_text(json.dumps(result, indent=2))

    def compute_aggregate_score(self) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregate scores from evaluation results.

        For each LLM, returns a dictionary with:
            - fcn_correct_pct: % of (task, llm) pairs with correct function implementations
            - avg_tests_passed_on_reference_pct: average % of test functions passed by reference impl
            - avg_expected_failures_detected_pct: average % of expected failures correctly failed

        Also saves each LLM's score dict to results_dir/llm_aggregate_{llm_name}.json
        """
        llm_metrics: Dict[str, Dict[str, float]] = {}

        # Intermediate counters
        counters: Dict[str, Dict[str, list]] = {}

        for task_id, llm_dict in self.results.items():
            for llm_name, eval_blocks in llm_dict.items():
                c = counters.setdefault(llm_name, {
                    "fcn_total": 0,
                    "fcn_correct": 0,
                    "test_pass_rates": [],
                    "failure_detection_rates": [],
                })

                # Count function correctness
                c["fcn_total"] += 1
                if eval_blocks.get("matches_reference", False):
                    c["fcn_correct"] += 1

                # Test-level results
                test_results = eval_blocks.get("tests", {}).get("test_results", {})
                ref_pass_pairs = test_results.get("reference_pass", [])
                ef_fail_pairs = test_results.get("failure_fail", [])

                if ref_pass_pairs:
                    n_passed = sum(1 for _, ok in ref_pass_pairs if ok)
                    c["test_pass_rates"].append(n_passed / len(ref_pass_pairs))

                if ef_fail_pairs:
                    n_failed = sum(1 for _, ok in ef_fail_pairs if ok)
                    c["failure_detection_rates"].append(n_failed / len(ef_fail_pairs))

        # Finalize scores per LLM
        for llm_name, c in counters.items():
            fcn_pct = 100.0 * c["fcn_correct"] / c["fcn_total"] if c["fcn_total"] > 0 else 0.0
            test_pct = 100.0 * sum(c["test_pass_rates"]) / len(c["test_pass_rates"]) if c["test_pass_rates"] else 0.0
            ef_pct = 100.0 * sum(c["failure_detection_rates"]) / len(c["failure_detection_rates"]) if c["failure_detection_rates"] else 0.0

            llm_metrics[llm_name] = {
                "fcn_correct_pct": round(fcn_pct, 2),
                "avg_tests_passed_on_reference_pct": round(test_pct, 2),
                "avg_expected_failures_detected_pct": round(ef_pct, 2),
            }

            # Write to disk
            out_path = self.results_dir / f"llm_aggregate_{llm_name}.json"
            out_path.write_text(json.dumps(llm_metrics[llm_name], indent=2))

        return llm_metrics

    def create_markdown_summary(self, filename="evaluation_summary.md"):
        """
        Save a Markdown summary of results across tasks and models.

        Creates two tables:
        1. Function correctness (✓ if matches_reference else ×)
        2. Test evaluation (percent passed on reference, percent of expected failures caught)

        The final row shows averages across all tasks for each model.
        """
        task_ids = sorted(self.results.keys())
        llm_names = sorted({llm for task in self.results.values() for llm in task})

        # Initialize tables
        code_table = []
        test_table_ref = []
        test_table_fail = []

        for task_id in task_ids:
            row_code = [task_id]
            row_ref = [task_id]
            row_fail = [task_id]

            for llm in llm_names:
                task_result = self.results.get(task_id, {}).get(llm, {})
                # Function correctness
                if task_result.get("matches_reference"):
                    row_code.append("✓")
                else:
                    row_code.append("×")

                # Test pass rate
                test_res = task_result.get("tests", {}).get("test_results", {})
                ref_pass = test_res.get("reference_pass", [])
                ef_fail = test_res.get("failure_fail", [])

                if ref_pass:
                    pct_pass = 100 * sum(ok for _, ok in ref_pass) / len(ref_pass)
                    row_ref.append(f"{pct_pass:.1f}%")
                else:
                    row_ref.append("–")

                if ef_fail:
                    pct_fail = 100 * sum(ok for _, ok in ef_fail) / len(ef_fail)
                    row_fail.append(f"{pct_fail:.1f}%")
                else:
                    row_fail.append("–")

            code_table.append(row_code)
            test_table_ref.append(row_ref)
            test_table_fail.append(row_fail)

        # Append total row
        total_row_code = ["Total"]
        total_row_ref = ["Avg Ref Pass %"]
        total_row_fail = ["Avg Fail Detect %"]

        for llm in llm_names:
            total = sum(1 for task in self.results if llm in self.results[task])
            correct = sum(
                1 for task in self.results
                if llm in self.results[task] and self.results[task][llm].get("matches_reference")
            )
            total_row_code.append(f"{correct}/{total}")

            ref_vals = []
            fail_vals = []
            for task in self.results:
                test_data = self.results[task].get(llm, {}).get("tests", {}).get("test_results", {})
                ref = test_data.get("reference_pass", [])
                fail = test_data.get("failure_fail", [])
                if ref:
                    ref_vals.append(100 * sum(ok for _, ok in ref) / len(ref))
                if fail:
                    fail_vals.append(100 * sum(ok for _, ok in fail) / len(fail))

            total_row_ref.append(f"{sum(ref_vals)/len(ref_vals):.1f}%" if ref_vals else "–")
            total_row_fail.append(f"{sum(fail_vals)/len(fail_vals):.1f}%" if fail_vals else "–")

        code_table.append(total_row_code)
        test_table_ref.append(total_row_ref)
        test_table_fail.append(total_row_fail)

        # Convert to Markdown
        def write_table(data, headers, title):
            df = pd.DataFrame(data, columns=headers)
            return f"### {title}\n\n" + df.to_markdown(index=False) + "\n\n"

        headers = ["Task"] + llm_names
        md = ""
        md += write_table(code_table, headers, "Function Correctness (✓ = Match)")
        md += write_table(test_table_ref, headers, "Reference Tests Passed (%)")
        md += write_table(test_table_fail, headers, "Expected Failures Detected (%)")

        # Save
        out_path = self.results_dir / filename
        out_path.write_text(md)