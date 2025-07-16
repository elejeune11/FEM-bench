from fem_bench.pipeline_utils import FEMBenchPipeline
from fem_bench.task_base import Task
import json
from pathlib import Path
import pytest
import tempfile


def test_fembench_pipeline_init():
    """Test initialization of FEMBenchPipeline and creation of required directories."""

    with tempfile.TemporaryDirectory() as tmp:
        tasks_dir = Path(tmp) / "tasks"
        llm_outputs_dir = Path(tmp) / "llm_outputs"
        prompts_dir = Path(tmp) / "prompts"
        results_dir = Path(tmp) / "results"

        pipeline = FEMBenchPipeline(
            tasks_dir=str(tasks_dir),
            llm_outputs_dir=str(llm_outputs_dir),
            prompts_dir=str(prompts_dir),
            results_dir=str(results_dir)
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

    # Instantiate and load
    pipeline = FEMBenchPipeline(
        tasks_dir=str(tasks_dir),
        llm_outputs_dir=str(llm_outputs_dir),
        prompts_dir=str(prompts_dir),
        results_dir=str(results_dir)
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

    pipeline = FEMBenchPipeline(
        tasks_dir=str(tasks_dir),
        llm_outputs_dir=str(llm_outputs_dir),
        prompts_dir=str(prompts_dir),
        results_dir=str(results_dir)
    )
    pipeline.load_all_tasks()
    pipeline.generate_and_save_task_prompts()

    for task_id in pipeline.tasks:
        prompt_file = prompts_dir / f"{task_id}_code_prompt.txt"
        assert prompt_file.exists(), f"Missing code prompt file for task '{task_id}'"
        content = prompt_file.read_text()
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

    pipeline = FEMBenchPipeline(
        tasks_dir=str(tasks_dir),
        llm_outputs_dir=str(llm_outputs_dir),
        prompts_dir=str(prompts_dir),
        results_dir=str(results_dir)
    )
    pipeline.load_all_tasks()
    pipeline.generate_and_save_test_prompts()

    for task_id in pipeline.tasks:
        prompt_file = prompts_dir / f"{task_id}_test_prompt.txt"
        assert prompt_file.exists(), f"Missing test prompt file for task '{task_id}'"
        content = prompt_file.read_text()
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
        results_dir=base_dir / "results_dir"
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
    parsed = json.loads(result_path.read_text())
    assert parsed == result_dict


def test_evaluate_all_llm_tests_real_example():
    base_dir = Path(__file__).parent
    pipeline = FEMBenchPipeline(
        tasks_dir=base_dir / "tasks_dir",
        llm_outputs_dir=base_dir / "llm_outputs_dir",
        prompts_dir=base_dir / "prompts_dir",
        results_dir=base_dir / "results_dir",
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
    parsed = json.loads(result_path.read_text())
    
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

    for llm_name, metrics in score_dict.items():
        assert isinstance(metrics, dict)
        assert set(metrics.keys()) == {
            "fcn_correct_pct",
            "avg_tests_passed_on_reference_pct",
            "avg_expected_failures_detected_pct"
        }

        # Confirm all values are floats
        for val in metrics.values():
            assert isinstance(val, float)

        # Confirm JSON file written and content matches
        file_path = pipeline.results_dir / f"llm_aggregate_{llm_name}.json"
        assert file_path.exists(), f"Missing output: {file_path}"
        file_metrics = json.loads(file_path.read_text())
        assert file_metrics == metrics