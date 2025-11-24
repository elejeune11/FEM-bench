from fem_bench.pipeline_utils import _json_default
from fem_bench.pipeline_utils import FEMBenchPipeline
from fem_bench.pipeline_utils import evaluate_task_tests as real_eval_tests
from fem_bench.pipeline_utils import validate_syntax
from fem_bench.task_base import Task
import json
import numpy as np
from pathlib import Path
import pytest
import tempfile
import textwrap


repo_root = Path(__file__).resolve().parents[1]

def test_json_default_handles_numpy_objects():
def test_json_default_is_robust_for_numpy_and_complex():
    obj = {
        # Arrays
        "arr_int": np.array([1, 2, 3], dtype=np.int32),
        "arr_float": np.array([1.5, 2.5, 3.5], dtype=np.float64),
        # Complex array mixes: real, near-real (tiny imag), and real+imag
        "arr_complex": np.array([1+0j, 2+1e-14j, 3+4j], dtype=np.complex128),

        # NumPy scalars
        "scalar_i": np.int32(7),
        "scalar_f": np.float64(3.14),
        "scalar_c_realish": np.complex128(5+1e-15j),   # should collapse to 5.0
        "scalar_c_complex": np.complex128(1+2j),       # should be {"real": 1.0, "imag": 2.0}

        # Python complex (not NumPy)
        "py_complex_realish": 10+1e-15j,               # should collapse to 10.0
        "py_complex": 0.5 - 3j,                        # dict form

        # Containers
        "as_tuple": (1, 2, 3),                         # -> [1,2,3]
        "as_set": {3, 1, 2},                           # -> [1,2,3] (sorted for determinism)

        # Nested structure
        "nested": {
            "inner_arr": np.array([0, 1]),
            "inner_tuple": (np.int64(9), np.float32(2.25)),
            "inner_complex_arr": np.array([1+0j, 0+2j]),
        },
    }

    # 1) Without default, at least one of these should fail
    with pytest.raises(TypeError):
        json.dumps(obj)

    # 2) With our helper, it should succeed and be JSON round-trippable
    dumped = json.dumps(obj, default=_json_default)
    loaded = json.loads(dumped)

    # --- Arrays ---
    assert loaded["arr_int"] == [1, 2, 3]
    assert loaded["arr_float"] == [pytest.approx(1.5), pytest.approx(2.5), pytest.approx(3.5)]

    # Complex array:
    #  - 1+0j -> 1.0
    #  - 2+1e-14j -> 2.0 (near-real collapse with tol 1e-12)
    #  - 3+4j -> {"real": 3.0, "imag": 4.0}
    assert loaded["arr_complex"][0] == pytest.approx(1.0)
    assert loaded["arr_complex"][1] == pytest.approx(2.0)
    assert loaded["arr_complex"][2] == {"real": 3.0, "imag": 4.0}

    # --- NumPy scalars ---
    assert loaded["scalar_i"] == 7
    assert loaded["scalar_f"] == pytest.approx(3.14)
    assert loaded["scalar_c_realish"] == pytest.approx(5.0)
    assert loaded["scalar_c_complex"] == {"real": 1.0, "imag": 2.0}

    # --- Python complex ---
    assert loaded["py_complex_realish"] == pytest.approx(10.0)
    assert loaded["py_complex"] == {"real": 0.5, "imag": -3.0}

    # --- Containers ---
    assert loaded["as_tuple"] == [1, 2, 3]
    assert loaded["as_set"] == [1, 2, 3]  # sorted for determinism

    # --- Nested structure checks ---
    assert loaded["nested"]["inner_arr"] == [0, 1]
    # (np.int64 -> int, np.float32 -> float)
    assert loaded["nested"]["inner_tuple"] == [9, pytest.approx(2.25)]
    # complex array inside nested:
    assert loaded["nested"]["inner_complex_arr"][0] == pytest.approx(1.0)
    assert loaded["nested"]["inner_complex_arr"][1] == {"real": 0.0, "imag": 2.0}


def test_fembench_pipeline_init():
    """Test initialization of FEMBenchPipeline and creation of required directories."""

    with tempfile.TemporaryDirectory() as tmp:
        tasks_dir = Path(tmp) / "tasks"
        llm_outputs_dir = Path(tmp) / "llm_outputs"
        prompts_dir = Path(tmp) / "prompts"
        results_dir = Path(tmp) / "results"
        template_dir = repo_root / "prompt_templates"
        

        pipeline = FEMBenchPipeline(
            tasks_dir=str(tasks_dir),
            llm_outputs_dir=str(llm_outputs_dir),
            prompts_dir=str(prompts_dir),
            results_dir=str(results_dir),
            prompt_template_dir=str(template_dir),
            code_prompt_template_name="code_prompt.j2",
            test_prompt_template_name="test_prompt.j2",
        )

        # Check directory paths are stored correctly
        assert pipeline.tasks_dir == tasks_dir
        assert pipeline.llm_outputs_dir == llm_outputs_dir
        assert pipeline.prompts_dir == prompts_dir
        assert pipeline.results_dir == results_dir

        # Check required directories exist
        assert prompts_dir.exists() and prompts_dir.is_dir()
        assert results_dir.exists() and results_dir.is_dir()

        # Check internal state is empty as expected
        assert pipeline.tasks == {}
        assert pipeline.prompts == {}
        assert pipeline.llm_outputs == {}
        assert pipeline.results == {}

        # Check repr output
        assert "FEMBenchPipeline" in repr(pipeline)
        assert "tasks=0" in repr(pipeline)


def test_load_tasks_from_local_tasks_dir():
    """
    Load all real task modules from 'tests/tasks_dir' and verify they return Task objects.
    """
    # Locate this test file's directory
    current_dir = Path(__file__).resolve().parent

    # Reference local tasks_dir under the tests directory
    tasks_dir = current_dir / "tasks_dir"
    llm_outputs_dir = current_dir / "llm_outputs_dir"
    prompts_dir = current_dir / "prompts_dir"
    results_dir = current_dir / "results_dir"
    template_dir = repo_root / "prompt_templates"

    # Instantiate and load
    pipeline = FEMBenchPipeline(
        tasks_dir=str(tasks_dir),
        llm_outputs_dir=str(llm_outputs_dir),
        prompts_dir=str(prompts_dir),
        results_dir=str(results_dir),
        prompt_template_dir=template_dir,
        code_prompt_template_name="code_prompt.j2",
        test_prompt_template_name="test_prompt.j2",
    )

    pipeline.load_all_tasks()

    # Sanity check
    assert len(pipeline.tasks) > 0, "No tasks loaded from tasks_dir"

    for task_id, task in pipeline.tasks.items():
        assert isinstance(task, Task), f"Task {task_id} is not a Task object"
        assert task.task_id == task_id
        assert task.main_fcn_code.strip().startswith("def"), f"Missing or malformed code in {task_id}"


def test_generate_and_save_task_prompts():
    """
    Generate and save code-generation prompts for real tasks in 'tests/tasks_dir'.
    """
    current_dir = Path(__file__).resolve().parent
    tasks_dir = current_dir / "tasks_dir"
    llm_outputs_dir = current_dir / "llm_outputs_dir"
    prompts_dir = current_dir / "prompts_dir"
    results_dir = current_dir / "results_dir"
    template_dir = repo_root / "prompt_templates"

    pipeline = FEMBenchPipeline(
        tasks_dir=str(tasks_dir),
        llm_outputs_dir=str(llm_outputs_dir),
        prompts_dir=str(prompts_dir),
        results_dir=str(results_dir),
        prompt_template_dir=template_dir,
        code_prompt_template_name="code_prompt.j2",
        test_prompt_template_name="test_prompt.j2",
    )
    pipeline.load_all_tasks()
    pipeline.generate_and_save_task_prompts()

    for task_id in pipeline.tasks:
        prompt_file = prompts_dir / f"{task_id}_code_prompt.txt"
        assert prompt_file.exists(), f"Missing code prompt file for task '{task_id}'"
        content = prompt_file.read_text(encoding="utf-8")
        assert "def " in content, "Prompt does not include function signature"
        assert "docstring" in content.lower(), "Prompt appears incomplete"
        assert "## Function Signature:" in content


def test_generate_and_save_test_prompts():
    """
    Generate and save test-generation prompts for real tasks in 'tests/tasks_dir'.
    """
    current_dir = Path(__file__).resolve().parent
    tasks_dir = current_dir / "tasks_dir"
    llm_outputs_dir = current_dir / "llm_outputs_dir"
    prompts_dir = current_dir / "prompts_dir"
    results_dir = current_dir / "results_dir"
    template_dir = repo_root / "prompt_templates"

    pipeline = FEMBenchPipeline(
        tasks_dir=str(tasks_dir),
        llm_outputs_dir=str(llm_outputs_dir),
        prompts_dir=str(prompts_dir),
        results_dir=str(results_dir),
        prompt_template_dir=template_dir,
        code_prompt_template_name="code_prompt.j2",
        test_prompt_template_name="test_prompt.j2",
    )
    pipeline.load_all_tasks()
    pipeline.generate_and_save_test_prompts()

    for task_id in pipeline.tasks:
        prompt_file = prompts_dir / f"{task_id}_test_prompt.txt"
        assert prompt_file.exists(), f"Missing test prompt file for task '{task_id}'"
        content = prompt_file.read_text(encoding="utf-8")
        assert "pytest" in content.lower(), "Prompt does not mention pytest"
        assert "## Test Functions to Implement:" in content
        assert "def " in content or "- " in content, "Prompt appears incomplete"


@pytest.fixture
def real_pipeline() -> FEMBenchPipeline:
    base_dir = Path(__file__).parent
    return FEMBenchPipeline(
        tasks_dir=base_dir / "tasks_dir",              # You can point to a dummy or real task dir
        llm_outputs_dir=base_dir / "llm_outputs_dir",  # <-- REAL directory with Gemini files
        prompts_dir=base_dir / "prompts_dir",
        results_dir=base_dir / "results_dir",
        prompt_template_dir=repo_root / "prompt_templates",
        code_prompt_template_name="code_prompt.j2",
        test_prompt_template_name="test_prompt.j2",
    )


def test_load_all_llm_outputs(real_pipeline):
    real_pipeline.load_all_llm_outputs()

    task_id = "element_stiffness_linear_elastic_1D"
    llm_name = "gemini"

    # ── Verify structure ──────────────────────────────────────────────
    assert task_id in real_pipeline.llm_outputs, f"{task_id} not found in llm_outputs"
    assert llm_name in real_pipeline.llm_outputs[task_id], f"{llm_name} not found for task {task_id}"

    llm_block = real_pipeline.llm_outputs[task_id][llm_name]

    # ── Check code block ──────────────────────────────────────────────
    assert "code" in llm_block, "Missing 'code' in LLM output"
    code_str = llm_block["code"]
    assert isinstance(code_str, str), "Code block must be a string"
    assert code_str.strip().startswith("def "), "Code must begin with a function definition"
    assert "import" not in code_str, "Code block should not contain import statements"

    # ── Check test block ──────────────────────────────────────────────
    assert "test" in llm_block, "Missing 'test' in LLM output"
    test_dict = llm_block["test"]
    assert isinstance(test_dict, dict), "Test block must be a dictionary"
    assert all(
        name.startswith("test_") for name in test_dict
    ), "All test keys should start with 'test_'"
    assert all(
        "def " in body for body in test_dict.values()
    ), "Each test function should include a function definition"


def test_evaluate_all_llm_outputs_real_example():
    base_dir = Path(__file__).parent
    pipeline = FEMBenchPipeline(
        tasks_dir=base_dir / "tasks_dir",
        llm_outputs_dir=base_dir / "llm_outputs_dir",
        prompts_dir=base_dir / "prompts_dir",
        results_dir=base_dir / "results_dir",
        prompt_template_dir=repo_root / "prompt_templates",
        code_prompt_template_name="code_prompt.j2",
        test_prompt_template_name="test_prompt.j2",
    )

    pipeline.load_all_tasks()
    pipeline.load_all_llm_outputs()
    pipeline.evaluate_all_llm_outputs()

    task_id = "element_stiffness_linear_elastic_1D"
    llm_name = "gemini"
    result_path = pipeline.results_dir / f"{task_id}_eval_{llm_name}.json"

    # ── Check results are present ─────────────────────────────
    assert task_id in pipeline.results
    assert llm_name in pipeline.results[task_id]
    result_dict = pipeline.results[task_id][llm_name]
    assert isinstance(result_dict, dict)
    assert "matches_reference" in result_dict

    # ── Check file written ────────────────────────────────────
    assert result_path.exists()
    parsed = json.loads(result_path.read_text(encoding="utf-8"))
    assert parsed == result_dict


def test_evaluate_all_llm_tests_real_example():
    base_dir = Path(__file__).parent
    pipeline = FEMBenchPipeline(
        tasks_dir=base_dir / "tasks_dir",
        llm_outputs_dir=base_dir / "llm_outputs_dir",
        prompts_dir=base_dir / "prompts_dir",
        results_dir=base_dir / "results_dir",
        prompt_template_dir=repo_root / "prompt_templates",
        code_prompt_template_name="code_prompt.j2",
        test_prompt_template_name="test_prompt.j2",
    )

    pipeline.load_all_tasks()
    pipeline.load_all_llm_outputs()
    pipeline.evaluate_all_llm_tests()

    task_id = "element_stiffness_linear_elastic_1D"
    llm_name = "gemini"
    result_path = pipeline.results_dir / f"{task_id}_tests_{llm_name}.json"

    # ── Validate results structure ─────────────────────────────
    assert task_id in pipeline.results
    assert llm_name in pipeline.results[task_id]
    result = pipeline.results[task_id][llm_name]
    assert "tests" in result

    test_block = result["tests"]
    assert test_block["task_id"] == task_id
    assert test_block["llm_name"] == llm_name
    assert test_block["tests_run"] is True
    assert "test_results" in test_block

    test_results = test_block["test_results"]
    assert isinstance(test_results, dict)
    assert "reference_pass" in test_results
    assert "failure_fail" in test_results

    # Validate contents of reference_pass
    assert isinstance(test_results["reference_pass"], list)
    for pair in test_results["reference_pass"]:
        assert isinstance(pair, (list, tuple)) and len(pair) == 2
        assert isinstance(pair[0], str) and isinstance(pair[1], bool)

    # ── Confirm file written and contents match ───────────────
    assert result_path.exists()
    parsed = json.loads(result_path.read_text(encoding="utf-8"))
    
    # Normalize tuples to lists for comparison
    def normalize(result_dict):
        for key in ("reference_pass", "failure_fail"):
            if key in result_dict["test_results"]:
                result_dict["test_results"][key] = [
                    list(item) for item in result_dict["test_results"][key]
                ]
        return result_dict

    assert normalize(parsed) == normalize(test_block)


def test_compute_aggregate_score_real_example():
    base_dir = Path(__file__).parent
    pipeline = FEMBenchPipeline(
        tasks_dir=base_dir / "tasks_dir",
        llm_outputs_dir=base_dir / "llm_outputs_dir",
        prompts_dir=base_dir / "prompts_dir",
        results_dir=base_dir / "results_dir",
        prompt_template_dir=repo_root / "prompt_templates",
        code_prompt_template_name="code_prompt.j2",
        test_prompt_template_name="test_prompt.j2",
    )

    # Load and evaluate
    pipeline.load_all_tasks()
    pipeline.load_all_llm_outputs()
    pipeline.evaluate_all_llm_outputs()
    pipeline.evaluate_all_llm_tests()

    # Run aggregation
    score_dict = pipeline.compute_aggregate_score()

    # Assert structure
    assert isinstance(score_dict, dict)
    assert len(score_dict) > 0, "No LLM scores returned"

    required_keys = {
        "fcn_correct_pct",
        "avg_tests_passed_on_reference_pct",
        "avg_expected_failures_detected_pct",
        "avg_joint_success_pct",  # new metric
    }

    for llm_name, metrics in score_dict.items():
        assert isinstance(metrics, dict)
        assert required_keys.issubset(metrics.keys()), (
            f"Missing expected keys in metrics for {llm_name}: "
            f"{required_keys - set(metrics.keys())}"
        )

        # Confirm all values are floats
        for val in metrics.values():
            assert isinstance(val, float)

        # Confirm JSON file written and content matches
        file_path = pipeline.results_dir / f"llm_aggregate_{llm_name}.json"
        assert file_path.exists(), f"Missing output: {file_path}"
        file_metrics = json.loads(file_path.read_text(encoding="utf-8"))
        assert file_metrics == metrics


def test_create_markdown_summary_real_example(real_pipeline):
    real_pipeline.load_all_tasks()
    real_pipeline.load_all_llm_outputs()
    real_pipeline.evaluate_all_llm_outputs()
    real_pipeline.evaluate_all_llm_tests()
    real_pipeline.compute_aggregate_score()
    real_pipeline.create_markdown_summary()

    summary_path = real_pipeline.results_dir / "evaluation_summary.md"
    assert summary_path.exists()
    text = summary_path.read_text(encoding="utf-8")

    # Updated headers
    assert "### Function Correctness" in text
    assert "### Joint Test Success Rate" in text

    # These should no longer be present
    assert "### Reference Tests Passed" not in text
    assert "### Expected Failures Detected" not in text


# -----------------------------------------------------------------------------
# Utility tests for static helper methods
# -----------------------------------------------------------------------------

def test_extract_function_code_success_and_failure():
    code_ok = textwrap.dedent(
        """
        import math

        def first(x):
            return x + 1

        def second(y):
            return y - 1
        """
    )
    assert FEMBenchPipeline.extract_function_code(code_ok).startswith("def first")

    # No function definition – should return None
    assert FEMBenchPipeline.extract_function_code("x = 42\n") is None


def test_load_all_llm_outputs_full_coverage(tmp_path: Path):
    # Create directory structure expected by FEMBenchPipeline
    tasks_dir = tmp_path / "tasks"
    prompts_dir = tmp_path / "prompts"
    llm_outputs_dir = tmp_path / "llm_outputs"
    results_dir = tmp_path / "results"
    for d in (tasks_dir, prompts_dir, llm_outputs_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Prepare files to hit every branch -----------------------------------

    # 1) CODE: SyntaxError
    (llm_outputs_dir / "task1_code_modelA.py").write_text("def bad(:\n  pass", encoding="utf-8")

    # 2) CODE: no function, but with a marker comment (triggers 'Marker:' path)
    (llm_outputs_dir / "task2_code_modelB.py").write_text(
        "# BLOCKED_SAFETY: provider=gemini-2.5-flash; reason=VIOLENCE\n", encoding="utf-8"
    )

    # 3) CODE: valid function (should be added to self.llm_outputs if allowed)
    (llm_outputs_dir / "task3_code_modelC.py").write_text(
        textwrap.dedent(
            """
            import math
            def foo(a, b):
                return a + b
            """
        ),
        encoding="utf-8",
    )

    # 4) TEST: SyntaxError
    (llm_outputs_dir / "task4_test_modelD.py").write_text("def test_x(:\n  pass", encoding="utf-8")

    # 5) TEST: no test_ functions, with marker comment (triggers 'Marker:' path)
    (llm_outputs_dir / "task5_test_modelE.py").write_text(
        "# BLOCKED_SAFETY: provider=gemini-2.5-flash; reason=VIOLENCE\n", encoding="utf-8"
    )

    # 6) TEST: valid test_ function (should be added to self.llm_outputs if allowed)
    (llm_outputs_dir / "task6_test_modelF.py").write_text(
        textwrap.dedent(
            """
            def test_example():
                assert 2 + 2 == 4
            """
        ),
        encoding="utf-8",
    )

    # 7) Non-matching file name (ignored)
    (llm_outputs_dir / "random.py").write_text("x = 42\n", encoding="utf-8")

    # Instantiate pipeline; tasks are irrelevant for this loader test
    pipe = FEMBenchPipeline(
        tasks_dir=str(tasks_dir),
        prompts_dir=str(prompts_dir),
        llm_outputs_dir=str(llm_outputs_dir),
        results_dir=str(results_dir),
        prompt_template_dir=repo_root / "prompt_templates",
        code_prompt_template_name="code_prompt.j2",
        test_prompt_template_name="test_prompt.j2",
    )

    # Only allow modelC and modelF to pass into self.llm_outputs
    pipe.load_all_llm_outputs(allowed_llms=["modelC", "modelF"])

    # -------------------- Assertions: results dict (skipped/error cases) --------------------

    # Task1/modelA => SyntaxError in function code
    r1 = pipe.results["task1"]["modelA"]
    assert r1["matches_reference"] is False
    assert "SyntaxError in function code" in r1["error"]

    # Task2/modelB => No function definition + Marker
    r2 = pipe.results["task2"]["modelB"]
    assert r2["matches_reference"] is False
    assert "No function definition found in the code." in r2["error"]
    assert "Marker: BLOCKED_SAFETY" in r2["error"]

    # Task4/modelD => SyntaxError in test code
    r4 = pipe.results["task4"]["modelD"]["tests"]
    assert r4["tests_run"] is False
    assert "SyntaxError in test code" in r4["error"]

    # Task5/modelE => No test functions found + Marker
    r5 = pipe.results["task5"]["modelE"]["tests"]
    assert r5["tests_run"] is False
    assert "No test functions found in the file." in r5["error"]
    assert "Marker: BLOCKED_SAFETY" in r5["error"]

    # -------------------- Assertions: llm_outputs (only allowed models included) ------------
    # Only modelC (code) and modelF (tests) should be present due to allowed_llms filter
    assert "task3" in pipe.llm_outputs
    assert "modelC" in pipe.llm_outputs["task3"]
    assert "code" in pipe.llm_outputs["task3"]["modelC"]
    assert pipe.llm_outputs["task3"]["modelC"]["code"].startswith("def foo(")

    assert "task6" in pipe.llm_outputs
    assert "modelF" in pipe.llm_outputs["task6"]
    assert "test" in pipe.llm_outputs["task6"]["modelF"]
    test_dict = pipe.llm_outputs["task6"]["modelF"]["test"]
    assert isinstance(test_dict, dict) and "test_example" in test_dict

    # Ensure non-allowed models are NOT in llm_outputs
    assert "task1" not in pipe.llm_outputs  # modelA is filtered (and syntax error)
    assert "task2" not in pipe.llm_outputs  # modelB is filtered (and no function)
    assert "task4" not in pipe.llm_outputs  # modelD is filtered (and syntax error)
    assert "task5" not in pipe.llm_outputs  # modelE is filtered (and no tests)

    # Non-matching 'random.py' should not produce any entries
    assert "random" not in pipe.llm_outputs


def _write_bad_task(py_path: Path):
    """Create a Python module with NO task_info() to trigger the error branch."""
    py_path.write_text("def foo():\n    return 0\n", encoding="utf-8")


def test_load_all_tasks_missing_task_info():
    with tempfile.TemporaryDirectory() as tmp:
        tasks_dir = Path(tmp) / "tasks"
        tasks_dir.mkdir()
        _write_bad_task(tasks_dir / "bad_task.py")

        pipeline = FEMBenchPipeline(
            tasks_dir=str(tasks_dir),
            llm_outputs_dir=str(Path(tmp) / "llm"),
            prompts_dir=str(Path(tmp) / "prompts"),
            results_dir=str(Path(tmp) / "results"),
            prompt_template_dir=repo_root / "prompt_templates",
            code_prompt_template_name="code_prompt.j2",
            test_prompt_template_name="test_prompt.j2",
        )

        with pytest.raises(ValueError):
            pipeline.load_all_tasks()


def test_load_all_llm_outputs_allowed_llms_filter():
    with tempfile.TemporaryDirectory() as tmp:
        llm_outputs = Path(tmp) / "llm_outputs"
        llm_outputs.mkdir()

        # Two fake LLM output files for the same (dummy) task
        (llm_outputs / "dummy_code_modelA.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (llm_outputs / "dummy_code_modelB.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

        pipeline = FEMBenchPipeline(
            tasks_dir=str(Path(tmp) / "tasks"),  # empty dir – fine for this test
            llm_outputs_dir=str(llm_outputs),
            prompts_dir=str(Path(tmp) / "prompts"),
            results_dir=str(Path(tmp) / "results"),
            prompt_template_dir=repo_root / "prompt_templates",
            code_prompt_template_name="code_prompt.j2",
            test_prompt_template_name="test_prompt.j2",
        )

        # Only include modelB
        pipeline.load_all_llm_outputs(allowed_llms=["modelB"])
        assert "dummy" in pipeline.llm_outputs
        assert "modelB" in pipeline.llm_outputs["dummy"]
        assert "modelA" not in pipeline.llm_outputs["dummy"]


def test_compute_aggregate_score_empty_results():
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = FEMBenchPipeline(
            tasks_dir=str(Path(tmp) / "tasks"),
            llm_outputs_dir=str(Path(tmp) / "llm"),
            prompts_dir=str(Path(tmp) / "prompts"),
            results_dir=str(Path(tmp) / "results"),
            prompt_template_dir=repo_root / "prompt_templates",
            code_prompt_template_name="code_prompt.j2",
            test_prompt_template_name="test_prompt.j2",
        )
        # No evaluation results added
        metrics = pipeline.compute_aggregate_score()
        assert metrics == {}, "Expected empty dict when no results are present"


def test_create_markdown_summary_with_no_results():
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp) / "results"
        pipeline = FEMBenchPipeline(
            tasks_dir=str(Path(tmp) / "tasks"),
            llm_outputs_dir=str(Path(tmp) / "llm"),
            prompts_dir=str(Path(tmp) / "prompts"),
            results_dir=str(results_dir),
            prompt_template_dir=repo_root / "prompt_templates",
            code_prompt_template_name="code_prompt.j2",
            test_prompt_template_name="test_prompt.j2",
        )

        # This should not raise even if results are empty
        pipeline.create_markdown_summary()
        summary_file = results_dir / "evaluation_summary.md"
        assert summary_file.exists()
        text = summary_file.read_text(encoding="utf-8")
        # With no tasks/models, the Markdown file will have headers only
        assert "### Function Correctness" in text


def test_extract_test_functions():
    code = textwrap.dedent(
        '''
        import pytest

        def helper():
            pass

        def test_alpha():
            'alpha'
            assert True

        def test_beta():
            assert 1 == 1
        '''
    )
    mapping = FEMBenchPipeline.extract_test_functions(code)
    assert set(mapping.keys()) == {"test_alpha", "test_beta"}
    for body in mapping.values():
        assert body.strip().startswith("def test_")


DUMMY_FCN_SRC = "def foo(x):\n    return x + 1\n"
DUMMY_TEST_SRC = "def test_foo(fcn):\n    assert fcn(1) == 2\n"

def _make_tmp_pipeline() -> FEMBenchPipeline:
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    return (
        FEMBenchPipeline(
            tasks_dir=str(base / "tasks"),
            llm_outputs_dir=str(base / "llm"),
            prompts_dir=str(base / "prompts"),
            results_dir=str(base / "results"),
            prompt_template_dir=repo_root / "prompt_templates",
            code_prompt_template_name="code_prompt.j2",
            test_prompt_template_name="test_prompt.j2",
        ),
        tmp,
    )


def test_extract_test_functions_empty_block():
    """Lines 137–138: branch where no test functions are present."""
    from fem_bench.pipeline_utils import FEMBenchPipeline
    code = "def helper():\n    pass\n"
    assert FEMBenchPipeline.extract_test_functions(code) == {}


def test_load_all_llm_outputs_parses_code_and_test_files():
    """Lines 146–148 & 214–215: file name parsing and LLM output structure."""
    pipeline, tmp = _make_tmp_pipeline()
    llm_dir = Path(pipeline.llm_outputs_dir)
    llm_dir.mkdir(parents=True, exist_ok=True)
    (llm_dir / "dummy_code_modelX.py").write_text(DUMMY_FCN_SRC, encoding="utf-8")
    (llm_dir / "dummy_test_modelX.py").write_text(DUMMY_TEST_SRC, encoding="utf-8")

    pipeline.load_all_llm_outputs()
    assert "dummy" in pipeline.llm_outputs
    assert "modelX" in pipeline.llm_outputs["dummy"]
    assert pipeline.llm_outputs["dummy"]["modelX"]["code"].startswith("def foo")
    assert "test_foo" in next(iter(pipeline.llm_outputs["dummy"]["modelX"]["test"].values()))
    tmp.cleanup()


def test_evaluate_all_llm_outputs_skips_unknown_task():
    """Lines 174–175: skipping unknown task_id."""
    pipeline, tmp = _make_tmp_pipeline()
    Path(pipeline.llm_outputs_dir).mkdir(parents=True, exist_ok=True)
    (Path(pipeline.llm_outputs_dir) / "unknown_code_modelZ.py").write_text(DUMMY_FCN_SRC, encoding="utf-8")

    pipeline.load_all_llm_outputs()
    pipeline.evaluate_all_llm_outputs()
    assert pipeline.results == {}
    tmp.cleanup()


def test_evaluate_all_llm_tests_unknown_task_is_ignored():
    """Lines 207–209: early return in evaluate_all_llm_tests."""
    pipeline, tmp = _make_tmp_pipeline()
    Path(pipeline.llm_outputs_dir).mkdir(parents=True, exist_ok=True)
    (Path(pipeline.llm_outputs_dir) / "ghost_code_modelY.py").write_text(DUMMY_FCN_SRC, encoding="utf-8")
    (Path(pipeline.llm_outputs_dir) / "ghost_test_modelY.py").write_text(DUMMY_TEST_SRC, encoding="utf-8")

    pipeline.load_all_llm_outputs()
    pipeline.evaluate_all_llm_tests()
    assert pipeline.results == {}
    tmp.cleanup()


def test_evaluate_all_llm_tests_handles_exception_and_sets_flag(monkeypatch):
    """Lines 255 and 276–278: internal error → tests_run = False."""
    pipeline, tmp = _make_tmp_pipeline()
    t = Task(
        task_id="simple",
        task_short_description="increment",
        created_date="2025‑07‑31",
        created_by="pytest",
        main_fcn_code=DUMMY_FCN_SRC,
        reference_verification_inputs=[[1]],
        test_cases=[{"test_code": DUMMY_TEST_SRC, "expected_failures": []}],
    )
    pipeline.tasks["simple"] = t

    Path(pipeline.llm_outputs_dir).mkdir(parents=True, exist_ok=True)
    (Path(pipeline.llm_outputs_dir) / "simple_code_modelQ.py").write_text(DUMMY_FCN_SRC, encoding="utf-8")
    (Path(pipeline.llm_outputs_dir) / "simple_test_modelQ.py").write_text(DUMMY_TEST_SRC, encoding="utf-8")
    pipeline.load_all_llm_outputs()

    def boom(*_a, **_kw):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("fem_bench.pipeline_utils.evaluate_task_tests", boom)

    pipeline.evaluate_all_llm_tests()
    res = pipeline.results["simple"]["modelQ"]["tests"]
    assert res["tests_run"] is False
    assert "simulated failure" in res["error"]
    tmp.cleanup()


def test_create_markdown_summary_totals_rows():
    """Check that totals and averages appear in the Markdown summary."""
    pipeline, tmp = _make_tmp_pipeline()
    pipeline.results = {
        "simple": {
            "llmA": {
                "matches_reference": False,
                "tests": {
                    "test_results": {
                        "reference_pass": [("t", True), ("t2", False)],
                        "failure_fail":  [("t", True), ("t2", True)]
                    }
                }
            }
        }
    }

    pipeline.create_markdown_summary()
    text = (Path(pipeline.results_dir) / "evaluation_summary.md").read_text(encoding="utf-8")

    # Check presence of totals and new metric header
    assert "Total" in text
    assert "Avg Joint Success %" in text

    # Old metrics no longer reported
    assert "Avg Ref Pass %" not in text
    assert "Avg Fail Detect %" not in text

    tmp.cleanup()



VALID_REF = "def inc(x):\n    return x + 1\n"
BROKEN_REF = "def broken("
RUNTIME_ERR_GEN = "def gen(x, y):\n    raise RuntimeError('boom')\n"
NEW_TEST = (
    "def test_new(fcn):\n"
    '    """Invented test (should trigger info print)."""\n'
    "    assert fcn(1) == 2\n"
)

def _make_pipeline(tmp_dir: Path) -> FEMBenchPipeline:
    return FEMBenchPipeline(
        tasks_dir=tmp_dir / "tasks",
        llm_outputs_dir=tmp_dir / "llm",
        prompts_dir=tmp_dir / "prompts",
        results_dir=tmp_dir / "results",
        prompt_template_dir=repo_root / "prompt_templates",
        code_prompt_template_name="code_prompt.j2",
        test_prompt_template_name="test_prompt.j2",
    )

def _register_task(pipeline, *, tid, code, tests):
    pipeline.tasks[tid] = Task(
        task_id=tid,
        task_short_description=f"{tid} description",
        created_date="2025‑07‑31",
        created_by="pytest",
        main_fcn_code=code,
        reference_verification_inputs=[[1], [2]],
        test_cases=tests,
    )


def test_pipeline_utils_all_remaining_branches():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        pipeline = _make_pipeline(base)

        _register_task(pipeline, tid="broken_task", code=BROKEN_REF, tests=[])
        _register_task(pipeline, tid="gen_error", code=VALID_REF, tests=[])
        _register_task(
            pipeline,
            tid="info_task",
            code=VALID_REF,
            tests=[{
                "test_code": (
                    "def test_existing(fcn):\n"
                    '    """Existing reference test."""\n'
                    "    assert fcn(2) == 3\n"
                ),
                "expected_failures": [],
            }],
        )
        _register_task(pipeline, tid="parse_err_task", code=VALID_REF, tests=[])

        llm_dir = base / "llm"
        llm_dir.mkdir(parents=True)
        (llm_dir / "broken_task_code_modelA.py").write_text("def foo():\n    return 0\n", encoding="utf-8")
        (llm_dir / "broken_task_test_modelA.py").write_text(NEW_TEST, encoding="utf-8")
        (llm_dir / "gen_error_code_modelA.py").write_text(RUNTIME_ERR_GEN, encoding="utf-8")
        (llm_dir / "gen_error_test_modelA.py").write_text(NEW_TEST, encoding="utf-8")
        (llm_dir / "info_task_code_modelA.py").write_text("def inc(x):\n    return x + 1\n", encoding="utf-8")
        (llm_dir / "info_task_test_modelA.py").write_text(NEW_TEST, encoding="utf-8")
        (llm_dir / "junk.py").write_text("pass\n", encoding="utf-8")

        pipeline.load_all_llm_outputs()
        pipeline.llm_outputs.setdefault("parse_err_task", {}).setdefault("modelA", {
            "code": VALID_REF,
            "test": {"bad": "def broken("},
        })

        pipeline.evaluate_all_llm_outputs()
        pipeline.evaluate_all_llm_tests()
        pipeline.create_markdown_summary()

        summary_md = (base / "results" / "evaluation_summary.md").read_text(encoding="utf-8")
        assert "Total" in summary_md
        assert "–" in summary_md 
        detail_errors = [
            d.get("error")
            for d in pipeline.results["gen_error"]["modelA"]["test_results"]
        ]
        assert any(err for err in detail_errors)
        assert "test_new" in str(pipeline.results["info_task"]["modelA"]["tests"]["test_results"])


VALID_REF_CODE = "def inc(x: int) -> int:\n    return x + 1\n"
BAD_GEN_CODE = '''
def bad(x):
    """RAISE_FLAG – forces loader failure"""
    return x + 1
'''


def _make_pipeline(tmp: Path) -> FEMBenchPipeline:
    return FEMBenchPipeline(
        tasks_dir=tmp / "tasks",
        llm_outputs_dir=tmp / "llm",
        prompts_dir=tmp / "prompts",
        results_dir=tmp / "results",
        prompt_template_dir=repo_root / "prompt_templates",
        code_prompt_template_name="code_prompt.j2",
        test_prompt_template_name="test_prompt.j2",
    )


def test_remaining_uncovered_lines(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        pipe = _make_pipeline(base)
        pipe.tasks["simple"] = __import__("types").SimpleNamespace(
            task_id="simple",
            main_fcn_code=VALID_REF_CODE,
            required_imports=[],
            fcn_dependency_code=[],
            reference_verification_inputs=[[1]],
            test_cases=[],
        )

        llm_dir = base / "llm"
        llm_dir.mkdir(parents=True)
        (llm_dir / "simple_code_syntaxErr.py").write_text(BAD_GEN_CODE, encoding="utf-8")
        (llm_dir / "simple_code_codeOnly.py").write_text(VALID_REF_CODE, encoding="utf-8")

        import fem_bench.pipeline_utils as pu
        original_loader = pu.load_function_from_code

        def patched(code, *a, **kw):
            if "RAISE_FLAG" in code:
                raise SyntaxError("simulated loader failure")
            return original_loader(code, *a, **kw)

        monkeypatch.setattr(pu, "load_function_from_code", patched)

        pipe.load_all_llm_outputs()
        pipe.evaluate_all_llm_outputs()
        pipe.evaluate_all_llm_tests()
        pipe.create_markdown_summary()

        # Validate internal results object
        assert "error" in pipe.results["simple"]["syntaxErr"]

        # Validate Markdown content
        md = (base / "results" / "evaluation_summary.md").read_text(encoding="utf-8")
        assert "### Function Correctness" in md
        assert "### Joint Test Success Rate" in md

        # Look for the totals line in joint metric table
        totals_line = next(line for line in md.splitlines() if line.startswith("| Avg Joint Success %"))
        assert "0.0%" in totals_line


def test_markdown_summary_triggers_ref_and_fail_detection():
    # Setup pipeline with dummy task + result
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pipe = FEMBenchPipeline(
            tasks_dir=tmp_path / "tasks",
            llm_outputs_dir=tmp_path / "llm",
            prompts_dir=tmp_path / "prompts",
            results_dir=tmp_path / "results",
            prompt_template_dir=repo_root / "prompt_templates",
            code_prompt_template_name="code_prompt.j2",
            test_prompt_template_name="test_prompt.j2",
        )
        pipe.results = {
            "dummy_task": {
                "llmA": {
                    "matches_reference": True,
                    "tests": {
                        "test_results": {
                            "reference_pass": [("test_x", True), ("test_y", False)],
                            "failure_fail": [("fail_1", True)],
                        }
                    }
                }
            }
        }

        # Call summary (which uses the two helpers)
        pipe.create_markdown_summary()

        # Confirm markdown file was written
        out_path = pipe.results_dir / "evaluation_summary.md"
        assert out_path.exists()
        text = out_path.read_text(encoding="utf-8")

        # Updated expectations
        assert "### Function Correctness" in text
        assert "### Joint Test Success Rate" in text
        assert "Avg Joint Success %" in text
        assert "0.0%" in text  # only one overlapping test ('test_x'), and test_y fails on ref



def test_compute_aggregate_score_handles_missing_test_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        pipe = FEMBenchPipeline(
            tasks_dir=base / "tasks",
            llm_outputs_dir=base / "llm",
            prompts_dir=base / "prompts",
            results_dir=base / "results",
            prompt_template_dir=repo_root / "prompt_templates",
            code_prompt_template_name="code_prompt.j2",
            test_prompt_template_name="test_prompt.j2",
        )

        # LLM output for one task, but NO test data
        pipe.results = {
            "dummy_task": {
                "llmA": {
                    "matches_reference": False,
                    # no "tests" key → should default to empty test set
                }
            }
        }

        scores = pipe.compute_aggregate_score()

        # Assert 0.0 was recorded for both test metrics
        assert "llmA" in scores
        assert scores["llmA"]["avg_tests_passed_on_reference_pct"] == 0.0
        assert scores["llmA"]["avg_expected_failures_detected_pct"] == 0.0


def test_valid_python_code():
    code = """
def add(a, b):
    return a + b
"""
    is_valid, error = validate_syntax(code)
    assert is_valid is True
    assert error is None


def test_invalid_assignment_expression():
    code = """
def broken():
    a + b = 3
"""
    is_valid, error = validate_syntax(code)
    assert is_valid is False
    assert isinstance(error, str)
    assert "cannot assign to expression" in error.lower()


def test_unclosed_string():
    code = """
def oops():
    return "missing end
"""
    is_valid, error = validate_syntax(code)
    assert is_valid is False
    assert "unterminated string" in error.lower() or "EOL while scanning" in error.lower()


def test_missing_colon():
    code = """
def bad(a, b)
    return a + b
"""
    is_valid, error = validate_syntax(code)
    assert is_valid is False
    assert "expected ':'" in error.lower()


def test_empty_string():
    is_valid, error = validate_syntax("")
    assert is_valid is True
    assert error is None


def test_load_all_llm_outputs_handles_syntax_errors(tmp_path):
    # Setup pipeline instance with temp dirs
    pipeline = FEMBenchPipeline(
        tasks_dir=tmp_path,
        llm_outputs_dir=tmp_path,
        prompts_dir=tmp_path,
        results_dir=tmp_path,
        prompt_template_dir=repo_root / "prompt_templates",
        code_prompt_template_name="code_prompt.j2",
        test_prompt_template_name="test_prompt.j2",
    )

    # ---- Create a function (code) file with bad syntax ----
    bad_code = "def broken():\n    a + b = 5"  # Invalid assignment
    code_file = tmp_path / "example_code_gptcode.py"
    code_file.write_text(bad_code, encoding="utf-8")

    # ---- Create a test file with bad syntax ----
    bad_test = "def test_fail(\n    assert True"  # Invalid syntax
    test_file = tmp_path / "example_test_gpttest.py"
    test_file.write_text(bad_test, encoding="utf-8")

    # ---- Run method under test ----
    pipeline.load_all_llm_outputs(allowed_llms=["gptcode", "gpttest"])

    # ---- Validate results for the code file ----
    code_result = pipeline.results.get("example", {}).get("gptcode", {})
    assert code_result["matches_reference"] is False
    assert "SyntaxError" in code_result["error"]

    # ---- Validate results for the test file ----
    test_result = pipeline.results.get("example", {}).get("gpttest", {}).get("tests", {})
    assert test_result["tests_run"] is False
    assert "SyntaxError" in test_result["error"]